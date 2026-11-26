import argparse
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import seaborn as sns

from simulation import (
    DEFAULT_ANNUAL_DRIFT,
    DEFAULT_ANNUAL_VOLATILITY,
    DEFAULT_DAYS,
    DEFAULT_NSTEPS,
    DEFAULT_TRADING_DAYS_PER_YEAR,
    require_code_directory,
    run_simulation,
    terminal_log_price_gaps,
)


FIGURES_DIR = '../Figures'


def plot_market_window(log_prices, daily_log_values, order_flow, signals, start_day, title):
    """Plot prices, true values and net order flow for a window of trading days."""
    days, nsteps = order_flow.shape
    steps = np.arange(1, days * nsteps + 1)
    prices = np.exp(log_prices[:, 1:]).ravel()
    true_values = np.repeat(np.exp(daily_log_values), nsteps)
    active = np.repeat(signals == 1, nsteps)

    fig, (price_ax, flow_ax) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
    )
    price_ax.set_title(title)
    price_ax.plot(steps, prices, linewidth=0.5, color='black', alpha=0.8, label='Market Price ($)')
    price_ax.plot(
        steps, np.where(active, np.nan, true_values),
        color='red', linestyle='-.', label='True Value (Inactive Insider)',
    )
    price_ax.plot(
        steps, np.where(active, true_values, np.nan),
        color='green', linestyle='--', label='True Value (Active Insider)',
    )
    price_ax.set_ylabel('Price ($)')
    price_ax.grid(True, alpha=0.25)

    flow_ax.plot(
        steps, order_flow.ravel(), color='dimgray', linewidth=0.25,
        alpha=0.8, label='Net Order Flow',
    )
    flow_ax.set_xlabel('Days')
    flow_ax.set_ylabel('Net Order Flow')
    flow_ax.grid(True, alpha=0.25)
    flow_ax.set_xticks(np.arange(days + 1) * nsteps)
    flow_ax.set_xticklabels(np.arange(start_day, start_day + days + 1))
    flow_ax.set_xlim(0, days * nsteps)

    price_handles, price_labels = price_ax.get_legend_handles_labels()
    flow_handles, flow_labels = flow_ax.get_legend_handles_labels()
    price_ax.legend(price_handles + flow_handles, price_labels + flow_labels, loc='best')
    fig.subplots_adjust(hspace=0.05)
    return fig


def plot_full_market_simulation(
    log_daily_closes, daily_log_values, signals,
    *, estimated=False, annual_drift=DEFAULT_ANNUAL_DRIFT,
    annual_volatility=DEFAULT_ANNUAL_VOLATILITY, days_in_year=DEFAULT_TRADING_DAYS_PER_YEAR,
):
    """Plot the full daily price path, including the opening price on day zero."""
    days = np.arange(len(daily_log_values) + 1)
    years = days / days_in_year
    log_open = log_daily_closes[0]
    active = np.asarray(signals) == 1
    true_values = np.exp(daily_log_values)
    expected_values = np.exp(log_open + annual_drift * years)
    log_median = log_open + (annual_drift - 0.5 * annual_volatility**2) * years
    log_band = NormalDist().inv_cdf(0.95) * annual_volatility * np.sqrt(years)
    value_label = 'Estimated True Value' if estimated else 'True Value'

    fig, price_ax = plt.subplots(figsize=(12, 4.2))
    price_ax.set_title(f'Full Market Simulation ({len(daily_log_values):,} Trading Days)')
    price_ax.plot(
        days, np.exp(log_daily_closes), linewidth=0.5, color='black', alpha=0.8,
        label='Market Close ($)',
    )
    # Connect the fundamental path across changes in insider activity.
    price_ax.plot(days, np.r_[np.exp(log_daily_closes[0]), true_values], color='grey', linewidth=0.4)
    price_ax.plot(
        days[1:], np.where(active, np.nan, true_values),
        color='red', linestyle='-.', marker='.', markersize=1.5, linewidth=0.8,
        label=f'{value_label} (Inactive Insider)',
    )
    price_ax.plot(
        days[1:], np.where(active, true_values, np.nan),
        color='green', linestyle='--', marker='.', markersize=1.5, linewidth=0.8,
        label=f'{value_label} (Active Insider)',
    )
    price_ax.plot(
        days, expected_values, color='tab:blue', linestyle=':', linewidth=1.3,
        label='Expected True Value (GBM)',
    )
    price_ax.fill_between(
        days, np.exp(log_median - log_band), np.exp(log_median + log_band),
        color='tab:blue', alpha=0.13, linewidth=0, zorder=0,
        label='Pointwise 5th to 95th Percentiles (GBM)',
    )
    price_ax.set_xlabel('Days')
    price_ax.set_ylabel('Price ($, log scale)')
    price_ax.set_yscale('log')
    price_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    price_ax.yaxis.set_major_formatter(formatter)
    price_ax.minorticks_off()
    price_ax.set_xlim(0, days[-1])
    price_ax.grid(True, alpha=0.25)
    price_ax.legend(loc='best')
    return fig


def plot_log_price_convergence(log_prices, daily_log_values, signals, limit):
    """Compare the mean absolute log-price gap by insider activity."""
    days = len(daily_log_values)
    nsteps = log_prices.shape[1] - 1
    indices = np.union1d(np.arange(limit), np.arange(days - limit, days))
    abs_diff = np.abs(log_prices[indices, 1:] - daily_log_values[indices, None])
    labels = np.where(signals[indices] == 1, 'Insider Active', 'Insider Inactive')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x=np.tile(np.arange(1, nsteps + 1), len(indices)),
        y=abs_diff.ravel(),
        hue=np.repeat(labels, nsteps),
        estimator='mean',
        errorbar='se',
        ax=ax,
    )
    ax.set_title(f'Log Price Convergence (First & Last {limit} Days)')
    ax.set_xlim(0, nsteps)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Absolute Difference |ln(Price) - ln(True Value)|')
    ax.axhline(0, color='black', linestyle='--', alpha=0.8)
    return fig


def plot_terminal_discrepancies(log_prices, daily_log_values, signals):
    """Compare terminal log-price gaps on active and inactive days."""
    active, inactive = terminal_log_price_gaps(log_prices, daily_log_values, signals)

    fig, ax = plt.subplots(figsize=(12, 6))
    for values, colour, alpha, label in (
        (inactive, 'blue', 0.25, 'Inactive Insider'),
        (active, 'red', 0.5, 'Active Insider'),
    ):
        if values.size >= 2 and np.ptp(values) > 0:
            sns.kdeplot(x=values, color=colour, fill=True, alpha=alpha, label=label, ax=ax)

    ax.set_title('Distribution of Terminal Log-Pricing Discrepancies (ln(Price) - ln(True Value))')
    ax.set_xlabel('Log-Price Discrepancy')
    ax.set_ylabel('Density')
    if inactive.size >= 2 and np.ptp(inactive) > 0:
        # Keep the broad inactive density visible alongside the narrow active peak.
        inactive_peak = 1 / (np.std(inactive) * np.sqrt(2 * np.pi))
        ax.set_ylim(0, 1.25 * inactive_peak)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(True, alpha=0.25)
    return fig


def save_simulation_figures(
    log_prices, daily_log_values, order_flow, signals,
    *, output_dir=FIGURES_DIR, show=False, annual_drift=DEFAULT_ANNUAL_DRIFT,
    annual_volatility=DEFAULT_ANNUAL_VOLATILITY, days_in_year=DEFAULT_TRADING_DAYS_PER_YEAR,
):
    """Export five PDF figures from a single simulation run."""
    days = len(daily_log_values)
    limit = min(20, days)
    figures = {
        'MarketSimulationFirstDays.pdf': plot_market_window(
            log_prices[:limit], daily_log_values[:limit], order_flow[:limit], signals[:limit],
            start_day=0, title=f'First {limit} Days of Market Simulation',
        ),
        'MarketSimulationFinalDays.pdf': plot_market_window(
            log_prices[-limit:], daily_log_values[-limit:], order_flow[-limit:], signals[-limit:],
            start_day=days - limit, title=f'Final {limit} Days of Market Simulation',
        ),
        'MarketSimulationFullPeriod.pdf': plot_full_market_simulation(
            np.r_[log_prices[0, 0], log_prices[:, -1]], daily_log_values, signals,
            annual_drift=annual_drift, annual_volatility=annual_volatility, days_in_year=days_in_year,
        ),
        'LogPriceConvergence.pdf': plot_log_price_convergence(
            log_prices, daily_log_values, signals, limit,
        ),
        'TerminalLogPricingDiscrepancies.pdf': plot_terminal_discrepancies(
            log_prices, daily_log_values, signals,
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for filename, fig in figures.items():
            fig.savefig(output_dir / filename, bbox_inches='tight')
        if show:
            plt.show()
    finally:
        for fig in figures.values():
            plt.close(fig)


def parse_run_args(description, *, full_pipeline=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help='Number of trading days.')
    parser.add_argument('--nsteps', type=int, default=DEFAULT_NSTEPS, help='Intraday steps per day.')
    parser.add_argument('--seed', type=int, help='Optional random seed for reproducing a simulation.')
    parser.add_argument('--show', action='store_true', help='Display the figures after saving them.')
    if full_pipeline:
        run_mode = parser.add_mutually_exclusive_group()
        run_mode.add_argument('--gemini', action='store_true', help='Run the original Gemini ETL and modelling agents.')
        run_mode.add_argument('--list-gemini-models', action='store_true', help='Show the Gemini model catalogue and exit.')
        run_mode.add_argument('--simulation-only', action='store_true', help='Export simulation figures, tables and intraday CSV.')
        parser.add_argument('--resume-modelling', action='store_true', help='Restart modelling from the saved daily summary.')
        model_choice = parser.add_mutually_exclusive_group()
        model_choice.add_argument('--gemini-model', metavar='MODEL', help='Use this Gemini model for the whole run.')
        model_choice.add_argument('--choose-model', action='store_true', help='Choose a Gemini model from a numbered list before the run.')
    args = parser.parse_args()
    if args.days < 1 or args.nsteps < 1:
        parser.error('--days and --nsteps must both be positive')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('--seed must be an integer from 0 to 4294967295')
    if full_pipeline and (args.gemini_model is not None or args.choose_model) and not args.gemini:
        parser.error('--gemini-model and --choose-model require --gemini')
    if full_pipeline and args.resume_modelling and not args.gemini:
        parser.error('--resume-modelling requires --gemini')
    require_code_directory()
    return args


def main():
    args = parse_run_args('Run the market simulation and export its figures.')
    if not args.show:
        plt.switch_backend('Agg')
    sim, daily_log_values, log_prices, order_flow, signals = run_simulation(
        args.days, args.nsteps, args.seed,
    )
    save_simulation_figures(
        log_prices, daily_log_values, order_flow, signals,
        show=args.show, annual_drift=sim.muP, annual_volatility=sim.sigmaP, days_in_year=sim.days_in_year,
    )
    print(f'Saved five PDFs to {FIGURES_DIR}')


if __name__ == '__main__':
    main()
