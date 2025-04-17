import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    def add_follow_ratio(self, df: pd.DataFrame,
                         followers_col: str = 'followers_count',
                         following_col: str = 'following_count') -> pd.DataFrame:
        """
        Добавляет признак: отношение числа подписчиков к числу подписок.

        :param df: датафрейм
        :return: датафрейм с новым признаком 'follow_ratio'
        """
        df['follow_ratio'] = df[followers_col] / (df[following_col] + 1e-5)
        return df

    def add_avg_posts_per_day(self, df: pd.DataFrame,
                               posts_col: str = 'statuses_count',
                               account_age_col: str = 'account_age_days') -> pd.DataFrame:
        """
        Добавляет признак: среднее количество постов в день.

        :param df: датафрейм
        :return: датафрейм с новым признаком 'avg_posts_per_day'
        """
        df['avg_posts_per_day'] = df[posts_col] / (df[account_age_col] + 1e-5)
        return df

    def engineer_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вызывает все функции генерации признаков.
        """
        df = self.add_follow_ratio(df)
        df = self.add_avg_posts_per_day(df)
        print("[INFO] Генерация новых признаков завершена.")
        return df
