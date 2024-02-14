import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, filepath):
        """
        Инициализирует обработчик данных, загружая данные из файла и выполняя начальную предобработку.

        :param filepath: путь к файлу с данными
        """
        self.df = self._load_data(filepath)
        self._initial_preprocessing()

    def _load_data(self, filepath):
        """
        Загружает данные из файла.

        :param filepath: путь к файлу с данными
        :return: датафрейм с данными
        """
        # Загрузка данных из файла
        data = pd.read_json(filepath)
        # Преобразование timestamp в datetime
        data['date'] = pd.to_datetime(data['date'], unit='ms')
        return data

    def _initial_preprocessing(self):
        """
        Выполняет начальную предобработку данных: устанавливает даты как индексы и интерполирует отсутствующие значения.
        """
        self.df.set_index('date', inplace=True)
        self.df = self.df[['rate']].interpolate()

    def get_episode_data(self, start_index, end_index):
        """
        Получает обработанные данные по эпизоду, конвертируя значения в процентное изменение относительно первого элемента.

        :param start_index: индекс начального элемента эпизода
        :param end_index: индекс конечного элемента эпизода
        :return: массив обработанных данных
        """
        episode = self.df.iloc[start_index:end_index]['rate'].to_numpy()
        base_value = episode[0]
        processed_episode = (episode - base_value) / base_value
        return processed_episode

    def get_datetime(self, index):
        """
        Возвращает дату и время для заданного индекса в строковом формате.

        :param index: индекс элемента
        :return: строковое представление даты и времени
        """
        datetime = self.df.index[index]
        return datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_price(self, index):
        """
        Возвращает цену для заданного индекса.

        :param index: индекс элемента
        :return: значение цены
        """
        current_rate = self.df.iloc[index]['rate']
        return current_rate

    def execute_trading_decision(self, index, stop_loss, take_profit):
        """
        Итерирует по данным, проверяя условия стоп-лосса и тейк-профита.

        :param index: индекс начального элемента
        :param stop_loss: значение стоп-лосса в процентах
        :param take_profit: значение тейк-профита в процентах
        :return: цена на момент срабатывания условия или -10000, если условие не сработало
        """
        initial_price = self.df.iloc[index]['rate']
        lower_bound = initial_price - initial_price * stop_loss / 100
        upper_bound = initial_price + initial_price * take_profit / 100

        for price in self.df.iloc[index:]['rate']:
            if price <= lower_bound or price >= upper_bound:
                return price
        return -10000

    def __len__(self):
        """
        Возвращает количество элементов в датафрейме.

        :return: количество элементов в датафрейме
        """
        return len(self.df)

