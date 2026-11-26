from pathlib import Path
from tempfile import TemporaryDirectory

from gemini_client import (
    RETRYABLE_STATUS_CODES, GeminiRequestError, generate_candidates, redact_gemini_text, upload_csvs,
)
from generate_simulation_tables import SUMMARY_FLOAT_FORMAT
from generated_code import run_generated_code
from simulation import LOGS_DIR


DAILY_FEATURE_COLUMNS = [
    'Day', 'Daily_Return', 'Price_Change', 'Mean_Log_Return', 'Return_Variance',
    'Return_Skewness', 'Return_Kurtosis', 'Mean_NetOrderFlow', 'NOF_Variance',
    'NOF_Skewness', 'NOF_Kurtosis',
]


ETL_PROMPT = '''

You are an AI agent for a financial regulator. You are tasked with taking an order flow dataset and summarizing into
a daily summary in the exact manner described below.

You are given two small CSV files drawn only from development days:
- 'df_sample.csv': up to 250 sampled intraday rows.
- 'df_metadata.csv': summary statistics and column types for the development data.


Use them only to infer column names, datatypes, and the meaning of fields.

Assumptions:
- Timestamp resets each day.
- 'Day' exists (if not, infer day boundaries from 'Timestamp' resets).
- 'Price ($)' is expressed in dollars.
- The 'Log Return' column is the log return from timestep to timestep.
- The 'NetOrderFlow' column is the net order flow from timestep to timestep. Net order flow is the sum of buy orders (+)
  and sell orders (-).

*** TASK: ***

Return two sections in this exact order:

[DECISION_LOG]
- List any additional assumptions about each metric (Day, Timestamp, Price ($), Log Return, NetOrderFlow) you make.
- Specify formulas in math notation or clear pseudocode.
- Explicitly state if you are using population or sample variances, and Fisher or Pearson kurtosis.
- Defend your justifications for choosing particular methods for calculating the statistical moments.
- List five sanity checks to validate the output.

[CODE]
Write Python (pandas/numpy/scipy allowed) code defining a function:

    def daily_summary(df) -> pd.DataFrame:

that returns one row per day with columns:
Day, Daily_Return, Price_Change, Mean_Log_Return, Return_Variance, Return_Skewness, Return_Kurtosis, 
Mean_NetOrderFlow, NOF_Variance, NOF_Skewness, NOF_Kurtosis

Ensure that you use the column names exactly as they are written above. You must use Program-of-Thought (Chen et al., 2023). 
Do not hard code decisions based on the provided samples, however. Instead, you should write "reasoning code" within your 
functions that dynamically assess the datasets at runtime.
Compute each day's features from that day's rows and the previous day's close. The input can start at any trading day.
Signal is kept separately by the workflow and is absent from the function's inputs and required outputs.

Column Descriptions:

- Day should correspond directly with Day, and will be used to identify the trading day.
- Daily_Return should be computed from Price ($), and should be measured from the closing price of the previous trading
  day to the closing price of the current trading day. For the first trading day, Daily_Return can be measured from the opening
  price of that day to the closing price. For consistency, this column should be expressed in log returns.
- Price_Change is the change in nominal price of the asset (i.e not expressed in a % like Daily_Return is).
  It should also be measured close-to-close like Daily_Return.
- Mean_Log_Return is based on 'Log Return' in the original dataset, and should be the mean of log return of all timesteps
  on a given trading day.
- Mean_NetOrderFlow is based on the 'NetOrderFlow' column in the original dataset, and should be the mean net order 
  flow of all timesteps on a given trading day.
- The Variance, Skewness, and Kurtosis calculations should be based on their respective distributions of 'Log Return' 
  and 'NetOrderFlow' (NOF). Values for'Log Return' and 'NetOrderFlow' should be taken for each timestep, NOT from their 
  average for a given trading day.

Warning/Hint: Note that in some versions of Pandas, .kurt() on groupby objects does not exist. It is recommended to
              use an alternative technique, such as .apply(lambda x: x.kurt()), instead.

Output only these two sections. Do not generate gratuitous text or code beyond what was requested unless absolutely
necessary.

'''


def aggregate_days(response, intraday, *, first_day, last_day, nsteps):
    """Apply the selected daily aggregation to a chronological window."""
    daily = run_generated_code(
        response, 'etl', intraday,
        rows=((first_day - 1) * nsteps, (last_day - first_day + 1) * nsteps),
    )
    missing = set(DAILY_FEATURE_COLUMNS) - set(daily)
    if missing:
        raise ValueError(f'Daily summary is missing columns: {", ".join(sorted(missing))}.')
    if daily['Day'].tolist() != list(range(first_day, last_day + 1)):
        raise ValueError('Daily summary must contain one row per input day in chronological order.')
    return daily[DAILY_FEATURE_COLUMNS]


def run_etl_agent(intraday, sample, metadata, *, days, nsteps, model, rate_limiter=None):
    """Generate and validate the ETL function on development data."""
    with TemporaryDirectory(prefix='insider-etl-context-') as directory:
        sample_path = Path(directory) / 'df_sample.csv'
        metadata_path = Path(directory) / 'df_metadata.csv'
        sample.to_csv(sample_path, index=False, float_format=SUMMARY_FLOAT_FORMAT)
        metadata.to_csv(metadata_path, index_label='feature', float_format=SUMMARY_FLOAT_FORMAT)
        files = upload_csvs((sample_path, metadata_path), model=model)
    prompt = ETL_PROMPT
    for attempt in range(1, 11):
        print(f'Running ETL agent. Attempt {attempt}:', flush=True)
        try:
            (response,) = generate_candidates(
                prompt, files, model=model, temperature=0, rate_limiter=rate_limiter,
            )
            daily = aggregate_days(response, intraday, first_day=1, last_day=days, nsteps=nsteps)
            break
        except GeminiRequestError as error:
            if error.status_code not in RETRYABLE_STATUS_CODES:
                raise
            print(str(error), flush=True)
        except Exception as error:
            message = redact_gemini_text(str(error))
            print(f'Failed with error: {message}', flush=True)
            prompt += (
                f'\n\n[SYSTEM ERROR]: The previous code you wrote failed with this error:'
                f'\n{message}\n\nPlease regenerate the code to fix this issue.'
            )
    else:
        raise GeminiRequestError('The ETL agent could not generate working code after 10 attempts.')

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / 'ETL_log.txt'
    log_path.write_text('=====ETL AGENT RESPONSE =====\n\n' + response + '\n', encoding='utf-8')
    print(f'Saved full ETL response to {log_path}')
    return daily, response
