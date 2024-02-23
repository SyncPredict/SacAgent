import gym
from gym import spaces
import numpy as np


class TradingEnv(gym.Env):
    """
    Оптимизированная среда для торговли фьючерсами Bitcoin. Позволяет агенту взаимодействовать с рынком,
    принимая решения о покупке, продаже и установке параметров стоп-лосс и тейк-профит на основе исторических данных о ценах.

    Attributes:
        data_processor (DataProcessor): Обработчик данных, предоставляющий исторические данные о ценах.
        window_size (int): Размер окна наблюдения, количество последних шагов, информация о которых доступна агенту.
        action_space (gym.spaces.Box): Пространство действий, определяющее возможные стоп-лосс и тейк-профит значения.
        observation_space (gym.spaces.Box): Пространство наблюдений, содержащее информацию о процентных изменениях цен в последнем окне.
        done (bool): Флаг завершения эпизода.
        current_step (int): Текущий шаг в рамках эпизода.
    """

    def __init__(self, data_processor, window_size=288):
        """
        Инициализирует среду торговли фьючерсами Bitcoin.

        :param data_processor: Экземпляр класса DataProcessor для обработки данных о ценах.
        :param window_size: Размер окна наблюдений (количество тиков), доступных агенту для принятия решений.
        """
        super(TradingEnv, self).__init__()
        self.done = None
        self.current_step = None
        self.data_processor = data_processor
        self.window_size = window_size
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size,),
                                            dtype=np.float32)
        self.reset()

    def __len__(self):
        return len(self.data_processor) - self.window_size

    def reset(self):
        """
        Сбрасывает среду к начальному состоянию для начала нового эпизода. Устанавливает текущий шаг (current_step)
        на значение, равное размеру окна наблюдений (window_size) плюс один. Это делается для корректного расчета
        процентного изменения, так как для первого наблюдения необходимо иметь предыдущее значение цены.

        :return: Начальное наблюдение среды - массив процентных изменений цен за последние window_size шагов,
                 рассчитанный начиная с элемента перед первым шагом в окне наблюдения.
        """
        self.current_step = self.window_size + 1
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        """
        Генерирует и возвращает текущее наблюдение среды, состоящее из процентного изменения цен за последние window_size шагов.
        Каждое процентное изменение рассчитывается относительно предыдущего значения цены, обеспечивая агенту последовательный
        временной ряд изменений цены для анализа.

        :return: Массив numpy, содержащий процентные изменения цен за интервал от current_step - window_size до current_step.
                 Пример возвращаемого значения для window_size=3: np.array([0.1, -0.0909, 0.0333]), где каждое значение отражает изменение
                 цены относительно предыдущего значения в последовательности.
        """
        obs = self.data_processor.get_episode_data(self.current_step - self.window_size, self.current_step)
        return obs

    def _take_action(self, action, current_price):
        """
        Выполняет действие, выбранное агентом, определяя изменение цены на основе заданных параметров стоп-лосс и тейк-профит.
        Функция анализирует, были ли достигнуты заданные уровни относительно текущей цены, и рассчитывает результат действия
        как процентное изменение от текущей цены до цены исполнения.

        :param action: Массив из двух элементов [стоп-лосс, тейк-профит], заданных агентом.
        :param current_price: Текущая цена актива на момент принятия решения.
        :return: Процентное изменение от текущей цены до цены исполнения, если были достигнуты условия стоп-лосса или тейк-профита.
                 Возвращает NaN, если условия не были достигнуты.
        """
        stop_loss, take_profit = action
        result_price = self.data_processor.execute_trading_decision(self.current_step, stop_loss, take_profit)
        if result_price == -10000:
            return float('nan')
        percentage_change = (result_price - current_price) / current_price
        return percentage_change

    def step(self, action):
        """
        Выполняет шаг в среде, применяя действие, выбранное агентом, и возвращает результат этого действия в виде нового наблюдения,
        награды, индикатора завершения эпизода и дополнительной информации.

        :param action: Действие, выбранное агентом, включающее в себя параметры стоп-лосс и тейк-профит.
        :return: Кортеж, содержащий следующее наблюдение (массив процентных изменений цен), награду (процентное изменение цены),
                 флаг завершения эпизода и дополнительную информацию (текущую дату и время).
        """
        current_price = self.data_processor.get_price(self.current_step)
        current_date = self.data_processor.get_datetime(self.current_step)
        self.current_step += 1

        reward = self._take_action(action, current_price)

        self.done = self.current_step >= len(self)

        obs = self._next_observation()
        return obs, reward, self.done, current_date
