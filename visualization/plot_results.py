import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set(style="whitegrid")


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Построение графика важности признаков для моделей с feature_importances_.

    :param model: обученная модель (например, RandomForest)
    :param feature_names: список имён признаков
    :param top_n: количество топовых признаков для отображения
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Топ {} важных признаков".format(top_n))
    plt.xlabel("Важность")
    plt.ylabel("Признаки")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba):
    """
    Построение ROC-кривой.

    :param y_true: истинные метки
    :param y_proba: вероятности предсказаний
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Кривая')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics_bar(metrics_dict):
    """
    Визуализация метрик в виде столбиковой диаграммы.

    :param metrics_dict: словарь с метриками
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), palette='deep')
    plt.title("Оценка качества модели")
    plt.ylabel("Значение")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
