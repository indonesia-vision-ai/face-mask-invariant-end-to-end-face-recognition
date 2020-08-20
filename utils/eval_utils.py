import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


def calculate_cosin_sim(x1, x2):
    similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return similarity


def calculate_acc_and_thresh(predictions, ground_truths):
    best_acc = 0
    best_threshold = 0

    total = len(predictions)

    for i in range(total):

        threshold = predictions[i]
        thresholded_predictions = (predictions >= threshold)
        acc = np.mean((thresholded_predictions == ground_truths).astype(int))

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_acc, best_threshold


def calculate_acc(predictions, ground_truths, threshold):
    thresholded_predictions = predictions >= threshold
    acc = np.mean((thresholded_predictions == ground_truths).astype(int))

    return acc


def roc_auc_threshold(predicted, target):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    roc_auc = auc(fpr, tpr)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({
        'tf' : pd.Series(tpr-(1-fpr), index=i),
        'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return fpr, tpr, roc_auc, list(roc_t['threshold'])