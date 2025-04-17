from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

def evaluate_model(y_true, y_pred, y_proba=None, average='binary'):
    """
    Выводит основные метрики классификации.

    :param y_true: истинные метки
    :param y_pred: предсказанные метки
    :param y_proba: вероятности предсказаний (для ROC-AUC)
    :param average: тип усреднения (binary, macro, micro)
    """
    print("[METRICS] Оценка модели:")
    print(f" - Accuracy      : {accuracy_score(y_true, y_pred):.4f}")
    print(f" - Precision     : {precision_score(y_true, y_pred, average=average):.4f}")
    print(f" - Recall        : {recall_score(y_true, y_pred, average=average):.4f}")
    print(f" - F1 Score      : {f1_score(y_true, y_pred, average=average):.4f}")

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            print(f" - ROC-AUC       : {auc:.4f}")
        except ValueError:
            print(" - ROC-AUC       : невозможно рассчитать (возможно, только один класс присутствует)")
    print("\n[REPORT]")
    print(classification_report(y_true, y_pred))


def get_metrics_dict(y_true, y_pred, y_proba=None, average='binary'):
    """
    Возвращает метрики в виде словаря.

    :param y_true: истинные метки
    :param y_pred: предсказанные метки
    :param y_proba: вероятности предсказаний
    :param average: тип усреднения
    :return: словарь метрик
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = None
    return metrics
