from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from generate_simulation_figures import FIGURES_DIR
from simulation import DATA_DIR


def save_confusion_matrix(actual, predicted, title, filename, *, show):
    matrix = confusion_matrix(actual, predicted, labels=[0, 1], normalize='true')
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=['Inactive Insider', 'Active Insider'],
    ).plot()
    display.ax_.set_title(title)
    display.figure_.savefig(Path(FIGURES_DIR) / filename, bbox_inches='tight')
    if not show:
        plt.close(display.figure_)


def evaluate_predictions(development, final_test, train_pred, test_pred, *, show=False):
    """Evaluate the predictions returned by the notebook's final model iteration."""
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for split, frame, predicted, title, filename in (
        ('development', development, train_pred, 'Agentic Model Training Set (Normalised)', 'DevelopmentConfusionMatrix.pdf'),
        ('final_test', final_test, test_pred, 'Agentic Model Testing Set (Normalised)', 'FinalTestConfusionMatrix.pdf'),
    ):
        actual = frame['Signal']
        results.append({
            'split': split,
            'accuracy': accuracy_score(actual, predicted),
            'precision': precision_score(actual, predicted, zero_division=0),
            'recall': recall_score(actual, predicted, zero_division=0),
            'f1': f1_score(actual, predicted, zero_division=0),
        })
        save_confusion_matrix(actual, predicted, title, filename, show=show)

    pd.DataFrame(results).to_csv(DATA_DIR / 'evaluation_metrics.csv', index=False)
    final_test[['Day', 'Signal']].assign(model_prediction=test_pred).to_csv(
        DATA_DIR / 'evaluation_predictions.csv', index=False,
    )
    print(f"Training Set Accuracy: {results[0]['accuracy']:.2%}")
    print(f"Testing Set Accuracy: {results[1]['accuracy']:.2%}")
    if show:
        plt.show()
        plt.close('all')
