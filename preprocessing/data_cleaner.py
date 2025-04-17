import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataCleaner:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет дубликаты из датафрейма.
        """
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        print(f"[INFO] Удалено дубликатов: {before - after}")
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Обрабатывает пропущенные значения.

        :param df: датафрейм
        :param strategy: 'mean', 'median' или 'drop'
        """
        if strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'drop':
            before = len(df)
            df = df.dropna()
            after = len(df)
            print(f"[INFO] Удалено строк с пропущенными значениями: {before - after}")
        else:
            raise ValueError("Стратегия должна быть 'mean', 'median' или 'drop'")
        return df

    def normalize_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Применяет MinMax-нормализацию к указанным признакам.

        :param df: датафрейм
        :param columns: список колонок для нормализации
        """
        df[columns] = self.scaler.fit_transform(df[columns])
        print(f"[INFO] Нормализованы колонки: {columns}")
        return df
