import tensorflow as tf
import numpy as np
from scipy.stats import rankdata


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def _binary_auc(y_true, y_score):
    """Compute AUC for binary classification using rank statistics."""
    pos = y_true == 1
    neg = y_true == 0
    pos_count = np.sum(pos)
    neg_count = np.sum(neg)
    if pos_count == 0 or neg_count == 0:
        return np.nan
    ranks = rankdata(y_score)
    sum_ranks_pos = np.sum(ranks[pos])
    return (sum_ranks_pos - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)


def masked_auc(preds, labels, mask):
    """AUC with masking (one-vs-rest averaged over classes)."""
    preds = np.array(preds)[mask]
    labels = np.array(labels)[mask]
    aucs = []
    for i in range(labels.shape[1]):
        auc = _binary_auc(labels[:, i], preds[:, i])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else np.nan


def _average_precision(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    cumsum = np.cumsum(y_true)
    precision = cumsum / (np.arange(len(y_true)) + 1)
    pos_total = np.sum(y_true)
    if pos_total == 0:
        return np.nan
    return float(np.sum(precision * y_true) / pos_total)


def masked_ap(preds, labels, mask):
    """Average precision with masking (one-vs-rest averaged over classes)."""
    preds = np.array(preds)[mask]
    labels = np.array(labels)[mask]
    aps = []
    for i in range(labels.shape[1]):
        ap = _average_precision(labels[:, i], preds[:, i])
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else np.nan


def masked_mrr(preds, labels, mask):
    """Mean Reciprocal Rank with masking."""
    preds = np.array(preds)[mask]
    labels = np.array(labels)[mask]
    rr = []
    for p, l in zip(preds, labels):
        true_idx = np.argmax(l)
        rank = np.where(np.argsort(-p) == true_idx)[0][0] + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr)) if rr else np.nan


def masked_hits_at_k(preds, labels, mask, k=100):
    """Hits@k with masking."""
    preds = np.array(preds)[mask]
    labels = np.array(labels)[mask]
    hits = []
    for p, l in zip(preds, labels):
        true_idx = np.argmax(l)
        top_k = np.argsort(-p)[:k]
        hits.append(1.0 if true_idx in top_k else 0.0)
    return float(np.mean(hits)) if hits else np.nan
