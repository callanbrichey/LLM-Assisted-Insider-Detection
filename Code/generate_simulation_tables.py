from pathlib import Path
import pandas as pd

from simulation import terminal_log_price_gaps


TABLES_DIR = '../Tables'
SUMMARY_FLOAT_FORMAT = '%.8g'


def create_metadata(df):
    """Summarise a dataset’s schema and values for the data agent."""
    describe = df.describe().T
    dtypes = df.dtypes.astype(str).rename('dtype')
    count = df.count().rename('non_null_count')
    return pd.concat([describe, dtypes, count], axis=1)


def sample_intraday(intraday):
    return intraday.sample(n=min(250, len(intraday)), random_state=12)


def save_simulation_tables(
    sim, log_prices, daily_log_values, order_flow, signals, *, output_dir=TABLES_DIR, intraday=None,
):
    """Write the terminal price diagnostic and full-simulation tables."""
    active, inactive = terminal_log_price_gaps(log_prices, daily_log_values, signals)
    gaps = pd.DataFrame({
        'Active Insider': pd.Series(active),
        'Inactive Insider': pd.Series(inactive),
    })
    gap_summary = gaps.describe().T.round(4).rename_axis('insider_status')

    if intraday is None:
        intraday = sim.dataset(log_prices, order_flow, signals)
    sample = sample_intraday(intraday)
    metadata = create_metadata(intraday)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_summary.to_csv(output_dir / 'TerminalLogPricingSummary.csv')
    sample_path, metadata_path = output_dir / 'df_sample.csv', output_dir / 'df_metadata.csv'
    sample.to_csv(sample_path, index=False)
    metadata.to_csv(metadata_path, index_label='feature', float_format=SUMMARY_FLOAT_FORMAT)
    return sample_path, metadata_path

