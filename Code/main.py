import json
from pathlib import Path
from time import perf_counter

import matplotlib
import pandas as pd

from etl_agent import DAILY_FEATURE_COLUMNS, aggregate_days, run_etl_agent
from eval import evaluate_predictions
from gemini_client import (
    DEFAULT_GEMINI_MODEL,
    PUBLISHED_FREE_TIER_MODELS,
    GeminiRequestError,
    GenerationRateLimiter,
    list_gemini_models,
    redact_gemini_text,
    resolve_gemini_model,
)
from generate_simulation_figures import FIGURES_DIR, parse_run_args, save_simulation_figures
from generate_simulation_tables import TABLES_DIR, create_metadata, sample_intraday, save_simulation_tables
from generated_code import run_generated_code
from modelling_agent import run_modelling_agent
from prepare_modelling_data import (
    FINAL_TEST_CSV, FINAL_TRAIN_CSV, prepare_modelling_data, save_final_test_data,
)
from simulation import DATA_DIR, INTRADAY_CSV, development_day_count, run_simulation


ETL_CHECKPOINT = Path('../_local/etl_checkpoint.json')


def show_gemini_models():
    models = list_gemini_models()
    if not models:
        raise GeminiRequestError('Google listed no Gemini models with generation support.')

    print('Gemini models that Google lists for content generation')
    for number, model in enumerate(models, start=1):
        if model == DEFAULT_GEMINI_MODEL:
            label = 'default, published free tier'
        elif model in PUBLISHED_FREE_TIER_MODELS:
            label = 'published free tier'
        else:
            label = 'check pricing'
        print(f'  {number}. {model} ({label})')
    print('Check model access and quota for your project at https://aistudio.google.com/rate-limit')
    return models


def choose_gemini_model():
    choices = {str(number): model for number, model in enumerate(show_gemini_models(), start=1)}
    while True:
        try:
            choice = input('Enter a model number, or Q to cancel: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit('Model selection cancelled.') from None
        if choice == 'q':
            raise SystemExit('Model selection cancelled.')
        if choice in choices:
            return choices[choice]
        print('Enter one of the listed numbers or Q.')


def simulate_and_aggregate(args, gemini_model, *, rate_limiter=None):
    simulation_started = perf_counter()
    print(f'Simulating {args.days:,} days at {args.nsteps:,} steps per day.', flush=True)
    sim, daily_log_values, log_prices, order_flow, signals = run_simulation(
        args.days, args.nsteps, args.seed,
    )
    intraday = sim.dataset(log_prices, order_flow, signals)
    save_simulation_tables(sim, log_prices, daily_log_values, order_flow, signals, intraday=intraday)
    if args.gemini:
        development_days = development_day_count(args.days)
        development_rows = intraday.iloc[:development_days * args.nsteps]
        sample = sample_intraday(development_rows).drop(columns='Signal')
        metadata = create_metadata(development_rows).drop(index='Signal')
        del development_rows
    INTRADAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    ETL_CHECKPOINT.unlink(missing_ok=True)
    intraday.to_csv(INTRADAY_CSV, index=False)
    del intraday
    print(f'Simulation elapsed time {perf_counter() - simulation_started:.2f} seconds.', flush=True)
    print(f'Saved three CSVs to {TABLES_DIR}')
    print(f'Saved the full intraday CSV to {INTRADAY_CSV}')
    save_simulation_figures(
        log_prices, daily_log_values, order_flow, signals,
        show=args.show, annual_drift=sim.muP, annual_volatility=sim.sigmaP, days_in_year=sim.days_in_year,
    )
    print(f'Saved five PDFs to {FIGURES_DIR}')
    if not args.gemini:
        return

    etl_started = perf_counter()
    try:
        development, response = run_etl_agent(
            INTRADAY_CSV, sample, metadata,
            days=development_days, nsteps=args.nsteps, model=gemini_model, rate_limiter=rate_limiter,
        )
    finally:
        print(f'ETL stage elapsed time {perf_counter() - etl_started:.2f} seconds.', flush=True)
    labels = signals.astype(int).tolist()
    development.assign(Signal=labels[:development_days]).to_csv(FINAL_TRAIN_CSV, index=False)
    source = INTRADAY_CSV.stat()
    checkpoint = {
        'response': response,
        'nsteps': args.nsteps,
        'signals': labels,
        'intraday_size': source.st_size,
        'intraday_mtime_ns': source.st_mtime_ns,
    }
    ETL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    pending = ETL_CHECKPOINT.with_suffix('.tmp')
    pending.write_text(json.dumps(checkpoint), encoding='utf-8')
    pending.replace(ETL_CHECKPOINT)
    print(f'Saved ETL output for {development_days:,} development days and a local resume checkpoint.')


def load_etl_checkpoint():
    try:
        checkpoint = json.loads(ETL_CHECKPOINT.read_text(encoding='utf-8'))
        development = pd.read_csv(FINAL_TRAIN_CSV, float_precision='round_trip')
        source = INTRADAY_CSV.stat()
        response = checkpoint['response']
        labels = checkpoint['signals']
        nsteps = checkpoint['nsteps']
        matching_source = (
            source.st_size == checkpoint['intraday_size']
            and source.st_mtime_ns == checkpoint['intraday_mtime_ns']
        )
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit('No readable ETL checkpoint. Start a full run with --gemini.') from None
    if (
        not matching_source or type(nsteps) is not int or nsteps < 1
        or not isinstance(response, str) or not response
        or not isinstance(labels, list) or len(labels) < 11 or any(label not in (0, 1) for label in labels)
        or len(development) != development_day_count(len(labels))
        or development.columns.tolist() != [*DAILY_FEATURE_COLUMNS, 'Signal']
        or development['Day'].tolist() != list(range(1, len(development) + 1))
        or development['Signal'].tolist() != labels[:len(development)]
    ):
        raise SystemExit('The ETL checkpoint does not match the saved inputs. Start a full run with --gemini.')
    return development, response, nsteps, labels


def main():
    args = parse_run_args('Run the synthetic market and insider detection workflow.', full_pipeline=True)
    if args.list_gemini_models:
        show_gemini_models()
        return
    if args.gemini and not args.resume_modelling and args.days < 11:
        raise SystemExit('Use at least 11 days for the notebook’s five modelling iterations.')
    gemini_model = None
    rate_limiter = None
    if args.gemini:
        gemini_model = choose_gemini_model() if args.choose_model else resolve_gemini_model(args.gemini_model)
        rate_limiter = GenerationRateLimiter()
        print(f'Selected Gemini model {gemini_model}.', flush=True)
    if not args.show:
        matplotlib.use('Agg')

    if not args.resume_modelling:
        simulate_and_aggregate(args, gemini_model, rate_limiter=rate_limiter)
    if not args.gemini:
        return

    development, etl_response, nsteps, labels = load_etl_checkpoint()
    if args.resume_modelling:
        print(f'Restarting modelling from saved ETL output ({len(development):,} development days).', flush=True)
    datasets = prepare_modelling_data(development)
    modelling_started = perf_counter()
    try:
        selected = run_modelling_agent(
            datasets, development, model=gemini_model, rate_limiter=rate_limiter,
        )
    finally:
        print(f'Modelling stage elapsed time {perf_counter() - modelling_started:.2f} seconds.', flush=True)

    print('Refitting the selected program on all development days and scoring the reserved days.', flush=True)
    try:
        # Include the last development day so the first test return uses its close.
        test_window = aggregate_days(
            etl_response, INTRADAY_CSV,
            first_day=len(development), last_day=len(labels), nsteps=nsteps,
        )
        test_features = test_window.iloc[1:]
        save_final_test_data(development, test_features)
        train, test = run_generated_code(selected, 'model', FINAL_TRAIN_CSV, FINAL_TEST_CSV)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f'Final scoring failed: {redact_gemini_text(str(error))}') from None
    train_pred, test_pred = train.to_numpy().ravel(), test.to_numpy().ravel()
    final_test = test_features.assign(Signal=labels[len(development):])
    daily = pd.concat([development, final_test], ignore_index=True)
    daily.to_csv(DATA_DIR / 'daily_summary.csv', index=False)
    evaluate_predictions(development, final_test, train_pred, test_pred, show=args.show)
    print(f'Saved modelling CSVs to {DATA_DIR}')


if __name__ == '__main__':
    try:
        main()
    except GeminiRequestError as error:
        raise SystemExit(str(error)) from None
