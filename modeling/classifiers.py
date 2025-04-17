from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import joblib

class ModelTrainer:
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Инициализирует модель по типу.

        :param model_type: 'random_forest', 'logistic_regression', 'gradient_boosting'
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._init_model()

    def _init_model(self) -> BaseEstimator:
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def train(self, X, y, test_size=0.2):
        """
        Обучает модель на данных.

        :param X: признаки
        :param y: целевая переменная
        :param test_size: размер тестовой выборки
        :return: кортеж (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        print(f"[INFO] Модель '{self.model_type}' обучена.")
        return X_train, X_test, y_train, y_test

    def predict(self, X):
        """
        Делает предсказания для новых данных.

        :param X: признаки
        :return: предсказанные значения
        """
        return self.model.predict(X)

    def save_model(self, filepath='model.pkl'):
        """
        Сохраняет обученную модель в файл.

        :param filepath: путь к файлу
        """
        joblib.dump(self.model, filepath)
        print(f"[INFO] Модель сохранена в '{filepath}'")

    def load_model(self, filepath='model.pkl'):
        """
        Загружает модель из файла.

        :param filepath: путь к файлу
        """
        self.model = joblib.load(filepath)
        print(f"[INFO] Модель загружена из '{filepath}'")
