import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from gemini_client import (
    RETRYABLE_STATUS_CODES, GeminiRequestError, GenerationRateLimiter,
    generate_candidates, redact_gemini_text, upload_csvs,
)
from generate_simulation_tables import SUMMARY_FLOAT_FORMAT
from generated_code import run_generated_code
from simulation import DATA_DIR, LOGS_DIR


def modelling_header(datasets):
    return f'''
*** SYSTEM INSTRUCTIONS: ***

You are an AI agent acting as a quantitative researcher for a financial regulator. You will be provided with the following:

- A random sample of up to 100 rows from each development training and validation split.
- A metadata summary of the full development training and validation splits for the current iteration.

Your task is to create a model that predicts whether or not a signal is active on a given trading day (i.e. the column labeled 'Signal').
The validation data have a blank Signal column; the workflow evaluates predictions separately.
The dataset you are provided is a summary of order flow by trading day; any given trading day has many individual time steps that were aggregated 
into the daily summary. The following are descriptions of the features in your dataset:

- Day: Identifies the trading day. Used for indexing, and thus should not be used as a feature of the model.
- Daily_Return: The log return of the trading day, measured close-to-close.
- Price_Change: The change in the asset's nominal value (not expressed as a percentage like Daily_Return). Like Daily_Return,
  it is also measured from close-to-close.
- Mean_Log_Return: The mean of the log return of all timesteps for a given trading day
- Return_Variance, Return_Skewness, Return_Kurtosis: The variance, skewness, and kurtosis of the distribution of
  log returns from timestep to timestep of a given trading day.
- Mean_NetOrderFlow: The mean net order flow of all timesteps for a given trading day.
- NOF_Variance, NOF_Skewness, NOF_Kurtosis: The variance, skewness, and kurtosis of the distribution of
  net order flows from timestep to timestep of a given trading day.
- Signal: The dependent variable, i.e. the one you are trying to predict.

For all of your responses, there are certain frameworks you must follow. When you are asked to reason, explain, deduce, etc., you
must use Chain-of-Thought (Wei et al., 2022) in your response. When you are asked to produce code, you must use Program-of-Thought
(Chen et al., 2023). Do not hard code decisions based on the provided samples, however. Instead, you should write "reasoning code"
within your functions that dynamically assess the datasets at runtime. There will be a sequence of {len(datasets)} prompts, following 
the Reflexion protocol (Shinn et al., 2023). In subsequent iterations after the first, you will have access to a comprehensive expanding 
memory. You are to use that memory to reason through your current instructions, reflect on previous responses and performance, and plan 
your current response with possible future iterations in mind, following the behavior of agents simulating human behavior in Park et al. (2023).

*** MASTER INSTRUCTIONS: ***

Return the TWO [DECISION_LOG] and [CODE] sections in this exact order:

[DECISION_LOG]

You are tasked with creating a binary classification model in Python that predicts the dependent variable 'Signal'.
The datasets you are provided are generated from a theoretical market microstructure model or models. Before generating your
predictive model, consider the following Step-Back prompt (Zheng et al., 2024): Identify what you believe that model or models to  
be, and why each feature you are given may be relevant to the underlying model(s). Include your answers to these questions in 
your response.

Now, you then must then specify how each of the provided features could be used to generate better informed classification 
predictions or for further feature engineering for that purpose, specifically in relation to the theoretical model(s). Your 
model must take into account the theoretical, conceptual, and mathematical properties of the market microstructure model(s) 
you believe the dataset to be generated from. There are no restrictions on packages or modeling approaches and techniques; 
you are free to use whatever approach you believe will have the highest out-of-sample predictive accuracy. Be sure to clearly 
state any preprocessing or feature engineering steps as well as any assumptions you make.

[CODE]

Write Python code defining two functions:

    def train_pred(df) -> pd.DataFrame:

    def test_pred(df) -> pd.DataFrame:

These functions should output single-column pandas DataFrames that contain your model's predictions on the train and test set 
respectively. The outputs must be binary class predictions, NOT probabilities. The file 'test_sample.csv' contains validation rows.
The workflow calls both functions locally on DataFrames. Fit the model and any data-dependent preprocessing in train_pred.
In test_pred, apply the fitted model and preprocessing using the same feature columns, including when scoring unseen days.
Keep the naming conventions. Do not read CSV files or convert the 
provided .csv files into strings within your code. Remember not to use 'Day' or "Signal' as independent variables in the model.

Remember to output only these two [DECISION_LOG] and [CODE] sections.
'''


def modelling_task(i):
    if i == 0:
        return f'''
        *** CURRENT TASK: ***

        Return TWO sections in this exact order:

        [DECISION_LOG]
        Refer to the instructions under *** MASTER INSTRUCTIONS: ***.
        
        [CODE]
        Refer to the instructions under *** MASTER INSTRUCTIONS: ***.

        Output only these two sections.
        '''
    return f'''
        Ignore previous task instructions, this is your new task.

        *** CURRENT TASK: ***
        
        Return TWO sections in this exact order:
        
        [DECISION_LOG]
        Recall ***SYSTEM INSTRUCTIONS: *** and your original [DECISION_LOG] instructions from *** MASTER INSTRUCTIONS: ***. 
        Reconsider the Step-Back prompt in *** MASTER INSTRUCTIONS: ***. Evaluate how everything in [HISTORY LOG: ITERATION {i}] 
        helps inform modeling decisions in the current iteration. Describe the conceptual, mathematical, and theoretical adaptations 
        you will make to the model in the current iteration to improve its out-of-sample predictive performance. Defend why you believe 
        these adaptations are good choices to improve your model.
        
        [CODE]
        Recall ***SYSTEM INSTRUCTIONS: *** and your original [CODE] instructions from *** MASTER INSTRUCTIONS: ***. Update your model, following 
        these instructions and trying to correct the flaws mentioned in the [DECISION_LOG] of this iteration. Reference the [HISTORY LOG: _] of 
        previous iterations if necessary. Recall that you are welcome to make any changes to the model as you see fit, including choosing an 
        entirely different approach. Your goal is simply to maximize predictive performance on out-of-sample data.
        '''


def get_metrics(train_true, train_pred, test_true, test_pred):
    
    ## Train metrics
    trn_accuracy = accuracy_score(train_true, train_pred)
    trn_precision = precision_score(train_true, train_pred)
    trn_recall = recall_score(train_true, train_pred)
    trn_f1 = f1_score(train_true, train_pred)
    trn_conf = confusion_matrix(train_true, train_pred)
    
    ## Validation metrics
    tst_accuracy = accuracy_score(test_true, test_pred)
    tst_precision = precision_score(test_true, test_pred)
    tst_recall = recall_score(test_true, test_pred)
    tst_f1 = f1_score(test_true, test_pred)
    tst_conf = confusion_matrix(test_true, test_pred)
    
    statement = f'''
    
    The following predictive measures are calculated from:
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    conf = confusion_matrix(true, pred)
    
    The predictive performance for your model on the seen training data of the last iteration:
    
    Accuracy: {trn_accuracy}
    Precision: {trn_precision}
    Recall: {trn_recall}
    f1: {trn_f1}
    Confusion Matrix (TN, FP, FN, TP): {trn_conf}
    
    The predictive performance for your model on the validation data of the last iteration:
    
    Accuracy: {tst_accuracy}
    Precision: {tst_precision}
    Recall: {tst_recall}
    f1: {tst_f1}
    Confusion Matrix [[TN, FP], [FN, TP]]: {tst_conf}
    '''
    return statement

def get_iter_msg(i, text_response, train_true, train_pred, test_true, test_pred, datasets):
    
    if i == len(datasets)-1:
        instructions = '''
        This is the final iteration. Consolidate your learnings or choose a new approach.
        Candidate selection still uses development validation data. The selected program will
        then be refit on all development days and evaluated on reserved days by the local workflow.
        '''
    elif i == len(datasets):
        instructions = 'This was the final iteration. Modeling complete.'
        
    else:
        instructions = 'You are now moving on to the next iteration.'
    
    iter_message = f'''
    
    ================================================================================
    [HISTORY LOG: ITERATION {i}]
    ================================================================================
    
    Below is the text response (including the code for the model) and the predictive 
    performance of the previous iteration, as well as your instructions for the next 
    iteration:
    
    [PREVIOUS AGENT RESPONSE & CODE]
    {text_response}
    
    [PREVIOUS PERFORMANCE EVALUATION]
    {get_metrics(train_true, train_pred, test_true, test_pred)}
    
    [INSTRUCTION FOR NEXT STEP]
    {instructions}
    
    ================================================================================
    '''
    
    return iter_message


def collect_valid_candidates(prompt, files, train, test, *, model, round_number, rate_limiter):
    """Fill five slots across up to five waves of parallel generation."""
    max_attempts = 5
    valid = {}
    last_request_error = None
    for attempt in range(1, max_attempts + 1):
        pending = [slot for slot in range(1, 6) if slot not in valid]
        print(f'Running modelling agent. Attempt {attempt}: requesting candidates {pending}.', flush=True)
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            requests = {
                slot: executor.submit(
                    generate_candidates, prompt, files, model=model, temperature=0.3, rate_limiter=rate_limiter,
                )
                for slot in pending
            }

        transient_failure = False
        code_errors = set()
        for slot, request in requests.items():
            try:
                (response,) = request.result()
            except GeminiRequestError as error:
                if error.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                last_request_error = error
                transient_failure = True
                print(f'Candidate {slot} in attempt {attempt} failed to generate: {error}', flush=True)
                continue

            try:
                train_out, test_out = run_generated_code(response, 'model', train, test)
                valid[slot] = (response, train_out.to_numpy().ravel(), test_out.to_numpy().ravel())
            except Exception as error:
                message = redact_gemini_text(str(error))
                code_errors.add(message)
                print(f'Candidate {slot} in attempt {attempt} failed validation: {message}', flush=True)

        print(f'Iteration {round_number}: {len(valid)} of 5 candidates have passed validation.', flush=True)
        if len(valid) == 5 or attempt == max_attempts:
            break
        if code_errors:
            error_log = '\n'.join(f'- {error}' for error in sorted(code_errors))
            prompt += (
                f'\n\n[SYSTEM ERROR]: Previous code candidates failed with the following error(s):\n{error_log}'
                '\nPlease regenerate the code to fix these issues.'
            )
        if transient_failure:
            delay = min(60, 30 * 2**(attempt - 1) + random.uniform(0, 3))
            print(f'Waiting {delay:.0f} seconds before retrying failed candidates.', flush=True)
            time.sleep(delay)

    if len(valid) < 3:
        message = (
            f'Iteration {round_number} stopped after {max_attempts} attempts with {len(valid)} of 5 valid candidates. '
            'At least 3 are required.'
        )
        if last_request_error is not None:
            raise GeminiRequestError(
                f'{message} Last API error: {last_request_error}', status_code=last_request_error.status_code,
            )
        raise GeminiRequestError(message)
    return [valid[slot] for slot in sorted(valid)]


def run_modelling_agent(datasets, development, *, model, rate_limiter=None):
    """Select a program through five rounds of development validation."""
    if rate_limiter is None:
        rate_limiter = GenerationRateLimiter()
    truth = development.set_index('Day')['Signal']
    memory = log = modelling_header(datasets)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (train, test, train_meta, test_meta) in enumerate(datasets):
        round_number = i + 1
        print(f'===== Iteration {round_number} =====', flush=True)
        current_prompt = modelling_task(i)
        train_sample = DATA_DIR / 'train_sample.csv'
        test_sample = DATA_DIR / 'test_sample.csv'
        train_df, test_df = pd.read_csv(train), pd.read_csv(test)
        for frame, path in ((train_df, train_sample), (test_df, test_sample)):
            frame.sample(min(100, len(frame)), random_state=12).to_csv(
                path, index=False, float_format=SUMMARY_FLOAT_FORMAT,
            )
        files = upload_csvs((train_sample, test_sample, train_meta, test_meta), model=model)
        time.sleep(3)

        full_prompt = memory + '\n' + current_prompt
        log += '\n' + current_prompt
        valid_candidates = collect_valid_candidates(
            full_prompt, files, train, test, model=model, round_number=round_number, rate_limiter=rate_limiter,
        )
        selection = f'Iteration {round_number}: calculating consensus from {len(valid_candidates)} valid candidates.'
        print(selection, flush=True)
        log += f'\n\n{selection}'
        consensus = np.mean([candidate[2] for candidate in valid_candidates], axis=0)
        response, train_pred, test_pred = min(
            valid_candidates, key=lambda candidate: np.sum((candidate[2] - consensus) ** 2),
        )
        print(f'Best candidate for iteration {round_number} chosen.')
        log_path = LOGS_DIR / f'iter{round_number}_response.txt'
        log_path.write_text(f'===== ITERATION {round_number} =====\n\n{response}\n', encoding='utf-8')
        print(f'Saved iteration response to {log_path}')
        history = get_iter_msg(
            round_number, response,
            truth.loc[train_df['Day']].to_numpy(), train_pred,
            truth.loc[test_df['Day']].to_numpy(), test_pred, datasets,
        )
        memory += history
        log += history
        time.sleep(15)

    log_path = LOGS_DIR / 'modelling_log.txt'
    log_path.write_text(log, encoding='utf-8')
    print(f'Saved full modelling log to {log_path}')
    return response
