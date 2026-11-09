import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def compute(y_true, prob_fake, threshold=0.5):
    y_true = np.asarray(y_true)
    prob_fake = np.asarray(prob_fake)
    pred = (prob_fake >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob_fake),
        "cm": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),  # sorok: valós 0/1, oszlopok: jósolt 0/1
    }
