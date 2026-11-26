import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def extract_code(response):
    match = re.search(r'\[CODE\]\s*(.*)', response, re.S)
    if not match:
        raise RuntimeError('Could not find [CODE] section in the Gemini response.')
    code = match.group(1).strip()
    fence = re.search(r'\x60\x60\x60(?:python)?\n(.*)\n\x60\x60\x60', code, re.S)
    return fence.group(1).strip() if fence else code


def run_generated_code(response, operation, *inputs, rows=None):
    """Run the notebook's generated functions in a child without the API key."""
    code = extract_code(response)
    # Keep the platform variables Python needs without forwarding credentials.
    platform_variables = {'PATH', 'SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'TEMP', 'TMP', 'LD_LIBRARY_PATH'}
    env = {name: value for name, value in os.environ.items() if name.upper() in platform_variables}
    worker = Path(__file__).resolve()
    with TemporaryDirectory(prefix='insider-agent-') as directory:
        scratch = Path(directory)
        code_path = scratch / 'model.py'
        code_path.write_text(code, encoding='utf-8')
        command = [sys.executable, '-I', '-B', str(worker), operation, str(code_path)]
        command.extend(str(Path(path).resolve()) for path in inputs)
        if rows is not None:
            command.extend(str(value) for value in rows)
        result = subprocess.run(command, cwd=scratch, env=env, capture_output=True)
        if result.returncode:
            lines = result.stderr.decode('utf-8', errors='replace').strip().splitlines()
            raise RuntimeError(lines[-1] if lines else f'Generated code exited with code {result.returncode}.')
        if operation == 'etl':
            return pd.read_csv(scratch / 'daily_summary.csv', float_precision='round_trip')
        return pd.read_csv(scratch / 'train_predictions.csv'), pd.read_csv(scratch / 'test_predictions.csv')


def run_worker(operation, code_path, *inputs):
    namespace = {'pd': pd, 'np': np}
    exec(compile(Path(code_path).read_text(encoding='utf-8'), code_path, 'exec'), namespace)
    if operation == 'etl':
        path, offset, count = inputs
        offset = int(offset)
        intraday = pd.read_csv(
            path,
            skiprows=None if offset == 0 else lambda row: 0 < row <= offset,
            nrows=int(count),
            usecols=lambda column: column != 'Signal',
            float_precision='round_trip',
        )
        daily = namespace['daily_summary'](intraday)
        daily.to_csv('daily_summary.csv', index=False)
    else:
        train, test = (pd.read_csv(path) for path in inputs)
        train_predictions = namespace['train_pred'](train)
        test_predictions = namespace['test_pred'](test)
        for predictions, frame, filename in (
            (train_predictions, train, 'train_predictions.csv'),
            (test_predictions, test, 'test_predictions.csv'),
        ):
            if not isinstance(predictions, pd.DataFrame) or predictions.shape != (len(frame), 1):
                raise ValueError('Each prediction function must return one DataFrame column with one row per day.')
            if not predictions.iloc[:, 0].isin((0, 1)).all():
                raise ValueError('Predictions must be binary labels (0 or 1).')
            predictions.to_csv(filename, index=False)


if __name__ == '__main__':
    run_worker(*sys.argv[1:])
