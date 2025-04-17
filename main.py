import pandas as pd
from data_loader.loader import load_data
from preprocessing.data_cleaner import clean_data
from preprocessing.feature_engineer import generate_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling.classifiers import RandomForestModel
from evaluation.metrics import get_metrics_dict
from visualization.plot_results import (
    plot_feature_importance,
    plot_roc_curve,
    plot_metrics_bar
)


def main():
    # 1. Загрузка данных
    filepath = 'data/users_vs_bots.csv'  # путь к CSV-файлу
    df = load_data(filepath)

    # 2. Предобработка
    df = clean_data(df)
    df = generate_features(df)

    # 3. Выбор признаков и целевой переменной
    target_column = 'label'  # имя целевой переменной
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Обучение модели
    model = RandomForestModel()
    model.train(X_train_scaled, y_train)

    # 7. Предсказания и оценка
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    metrics = get_metrics_dict(y_test, y_pred, y_proba)

    # 8. Вывод метрик
    print("\nОценка модели:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # 9. Визуализация
    plot_feature_importance(model.model, X.columns)
    plot_roc_curve(y_test, y_proba)
    plot_metrics_bar(metrics)


if __name__ == '__main__':
    main()
