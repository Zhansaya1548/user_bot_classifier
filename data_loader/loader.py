import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path: str):
        """
        Инициализация загрузчика данных.

        :param file_path: Путь к CSV-файлу с данными
        """
        self.file_path = "bots_vs_users.csv"
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Загружает данные из CSV-файла.

        :return: DataFrame с загруженными данными
        :raises FileNotFoundError: если файл не найден
        :raises ValueError: если файл не содержит данных
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Файл по пути '{self.file_path}' не найден.")

        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Ошибка при чтении файла: {e}")

        if self.data.empty:
            raise ValueError("Файл загружен, но он пустой.")

        print(f"[INFO] Данные успешно загружены из '{self.file_path}' (строк: {len(self.data)}, колонок: {len(self.data.columns)})")
        return self.data

    def preview_data(self, n: int = 5) -> pd.DataFrame:
        """
        Показывает первые n строк данных.

        :param n: Количество строк для отображения
        :return: DataFrame с первыми n строками
        """
        if self.data is None:
            raise ValueError("Данные ещё не загружены. Вызовите load_data() сначала.")
        return self.data.head(n)
