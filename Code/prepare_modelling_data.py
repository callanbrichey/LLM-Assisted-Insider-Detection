import numpy as np

from generate_simulation_tables import SUMMARY_FLOAT_FORMAT, create_metadata
from simulation import DATA_DIR


FINAL_TRAIN_CSV = DATA_DIR / 'agent_final_train.csv'
FINAL_TEST_CSV = DATA_DIR / 'agent_final_test.csv'


def prepare_modelling_data(development):
    """Create the expanding validation splits and reuse the last for selection."""
    datasets = []

    def save_split(train, test, train_name, test_name, train_metadata, test_metadata):
        if train.empty or test.empty or train['Day'].isin(test['Day']).any():
            raise ValueError('A modelling split is empty or has overlapping train and test days.')
        test_for_agent = test.assign(Signal=np.nan)
        train_path, test_path = DATA_DIR / train_name, DATA_DIR / test_name
        train_meta_path, test_meta_path = DATA_DIR / train_metadata, DATA_DIR / test_metadata
        train.to_csv(train_path, index=False)
        test_for_agent.to_csv(test_path, index=False)
        create_metadata(train).to_csv(train_meta_path, float_format=SUMMARY_FLOAT_FORMAT)
        create_metadata(test_for_agent).to_csv(test_meta_path, float_format=SUMMARY_FLOAT_FORMAT)
        return train_path, test_path, train_meta_path, test_meta_path

    for i, fraction in enumerate(np.linspace(0.25, 1, 4), start=1):
        current = development.iloc[:int(np.floor(fraction * len(development)))]
        current.to_csv(DATA_DIR / f'iter{i}.csv', index=False)
        boundary = int(len(current) * 0.8)
        datasets.append(save_split(
            current.iloc[:boundary], current.iloc[boundary:],
            f'iter{i}_train.csv', f'iter{i}_test.csv', f'iter{i}_train_meta.csv', f'iter{i}_test_meta.csv',
        ))

    datasets.append(datasets[-1])
    return datasets


def save_final_test_data(development, test_features):
    """Save local scoring inputs and their metadata."""
    test_input = test_features.assign(Signal=np.nan)
    test_input.to_csv(FINAL_TEST_CSV, index=False)
    create_metadata(development).to_csv(
        DATA_DIR / 'agent_final_train_metadata.csv', float_format=SUMMARY_FLOAT_FORMAT,
    )
    create_metadata(test_input).to_csv(
        DATA_DIR / 'agent_final_test_metadata.csv', float_format=SUMMARY_FLOAT_FORMAT,
    )
