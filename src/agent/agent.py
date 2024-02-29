import numpy as np
import gc  # Импортируем модуль сборщика мусора

import pandas as pd

from src.agent.sac import SACAgent, clear_memory
import wandb

from src.utils.dir import find_directory
from src.utils.visualize import print_progress
from src.trading_env.env import TradingEnv


class Agent(SACAgent):
    """
    Агент, использующий алгоритм Soft Actor-Critic для оптимизации торговой стратегии в среде фьючерсов Bitcoin.

    Attributes:
        state_dim (int): Размерность состояний среды.
        action_dim (int): Размерность действий агента.
        hidden_dim (int): Размерность скрытого слоя в сетях агента.
        replay_buffer (ReplayBuffer): Буфер воспроизведения для хранения опыта агента.
        batch_size (int): Размер батча для обучения.
        lr (float): Скорость обучения.
        gamma (float): Коэффициент дисконтирования.
        tau (float): Коэффициент мягкого обновления целевых сетей.
        alpha (float): Температурный параметр для регулирования trade-off между исследованием и эксплуатацией.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        replay_buffer,
        batch_size,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    ):
        """
        Инициализирует агента с заданными параметрами и настройками для обучения и взаимодействия с средой.

        :param state_dim: Размерность вектора состояния.
        :param action_dim: Размерность вектора действия.
        :param hidden_dim: Размер скрытых слоёв в архитектуре нейронной сети.
        :param replay_buffer: Экземпляр ReplayBuffer для хранения и воспроизведения опыта.
        :param batch_size: Размер батча для обучения.
        :param lr: Скорость обучения.
        :param gamma: Фактор дисконтирования будущих наград.
        :param tau: Коэффициент мягкого обновления для целевых сетей.
        :param alpha: Параметр, контролирующий компромисс между исследованием и эксплуатацией.
        """
        super().__init__(state_dim, action_dim, hidden_dim, lr, gamma, tau, alpha)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def execute_episodes(self, env, episodes=20, train=False):
        """
        Выполняет заданное количество эпизодов в среде, опционально обновляя параметры агента.

        :param env: Среда, в которой агент будет действовать.
        :param episodes: Количество эпизодов для выполнения.
        :param train: Флаг, определяющий, будет ли агент обучаться на основе выполненных эпизодов.
        """
        new_dir = find_directory(True)
        for episode in range(episodes):
            rewards = self._process_episode(env, episode, episodes, train, self.batch_size)
            total_rewards = sum(rewards)  # Общая награда за эпизод
            average_reward_per_step = (
                total_rewards / len(rewards)
            )  # Средняя награда за шаг
            positive_rewards_count = sum([1 for reward in rewards if reward > 0])  # Количество положительных наград
            win_rate = positive_rewards_count / len(rewards) * 100  # Процент положительных наград

            # Логирование метрик
            wandb.log(
                {
                    "Episode total rewards": total_rewards,
                    "Average reward per step": average_reward_per_step,
                    "Win rate (%)": win_rate,
                }
            )

            if train:
                self.update_parameters(self.replay_buffer, self.batch_size)
                self.replay_buffer.reset()
                self.save_model(new_dir, f"episode_{episode}")

            clear_memory()
            gc.collect()

    def _process_episode(self, env: TradingEnv, episode, episodes, train, batch_size):
        """
        Обрабатывает один эпизод, собирая данные о действиях, наградах и датах.
        Возвращает результаты в формате pandas DataFrame, где дата является индексом,
        а действия и награды представлены в столбцах.

        :param env: Среда, в которой агент действует.
        :param episode: Текущий номер эпизода.
        :param episodes: Общее количество эпизодов.
        :param train: Флаг, определяющий, будет ли буфер заполняться.
        :return: DataFrame с колонками ['action', 'reward'] и датами в качестве индекса.

        Пример выходных параметров:
        |    date    |  action  |  reward  |
        |------------|----------|----------|
        | 2021-01-01 |    0.5   |    10    |
        | 2021-01-02 |    0.6   |    15    |
        """
        states = env.reset()

        rewards = []
        done = False
        while not done:
            actions = self.select_actions(states)
            next_states, result, done, _ = env.step(actions)

            if done:
                break

            for i in range(len(result)):
                action, reward = result[i]
                prev_window = states[i]
                next_window = next_states[0] if i == len(result) - 1 else states[i + 1]
                if not np.isnan(reward) and train:
                    self.replay_buffer.add(
                        (prev_window, action, next_window, reward, float(done))
                    )  # Добавление в буфер

                rewards.append(reward)

            states = next_states
            print_progress(env.current_step, env.max_steps, episode, episodes, sum(rewards))

        return rewards
