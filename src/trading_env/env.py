import gym
from gym import spaces
import numpy as np


class TradingEnv(gym.Env):
    """Оптимизированная среда для торговли фьючерсами Bitcoin."""

    def __init__(self, data_processor, window_size=288):
        super(TradingEnv, self).__init__()  # Исправлено название класса
        self.done = None
        self.current_step = None
        self.data_processor = data_processor
        self.window_size = window_size
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),  # Изменено на нижний предел
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size,),
                                            dtype=np.float32)  # Изменен shape
        self.reset()

    def __len__(self):
        return len(self.data_processor) - self.window_size

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        obs = self.data_processor.get_episode_data(self.current_step - self.window_size, self.current_step)
        return obs  # Возвращается уже в обработанном виде

    def _take_action(self, action, current_price):
        stop_loss, take_profit = action  # Использование нового метода
        # Пересчет цен не требуется, используем функцию check_stop_loss_take_profit
        result_price = self.data_processor.execute_trading_decision(self.current_step, stop_loss, take_profit)
        if result_price == -10000:
            return float('nan')
        percentage_change = (result_price - current_price) / current_price
        return percentage_change

    def step(self, action):
        current_price = self.data_processor.get_price(self.current_step)
        current_date = self.data_processor.get_datetime(self.current_step)
        self.current_step += 1

        reward = self._take_action(action, current_price)

        self.done = self.current_step >= len(self)

        obs = self._next_observation()
        return obs, reward, self.done, current_date
