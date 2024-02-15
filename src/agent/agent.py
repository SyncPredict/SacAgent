import json

import numpy as np
import gc  # Импортируем модуль сборщика мусора

from src.agent.sac import SACAgent
import wandb
import os

from src.utils.dir import find_directory
from src.utils.visualize import print_progress


class Agent(SACAgent):

    def __init__(self, state_dim, action_dim, hidden_dim, replay_buffer, batch_size, lr=1e-4, gamma=0.99, tau=0.005,
                 alpha=0.2):
        super().__init__(state_dim, action_dim, hidden_dim, lr, gamma, tau, alpha)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def execute_episodes(self, env, episodes=20, train=False):
        new_dir = find_directory(True)
        for episode in range(episodes):
            rewards, actions, dates = self._process_episode(env, episode, episodes, train)
            total_rewards = sum(rewards)  # Общая награда за эпизод
            average_reward_per_step = total_rewards / len(
                rewards) if rewards else 0  # Средняя награда за шаг, предотвращаем деление на 0
            positive_rewards_count = len([r for r in rewards if r > 0])  # Количество положительных наград
            win_rate = (positive_rewards_count / len(
                rewards) * 100) if rewards else 0  # Процент положительных наград, предотвращаем деление на 0

            # Логирование метрик
            wandb.log({
                'Episode total rewards': total_rewards,
                'Average reward per step': average_reward_per_step,
                'Win rate (%)': win_rate,
                    'Rewards distribution': wandb.Table(rewards)  # Гистограмма распределения наград за шаги
            })

            if train:
                self.update_parameters(self.replay_buffer, self.batch_size)
                self.replay_buffer.reset()
                self.save_model(new_dir, f'episode_{episode}')

            gc.collect()

    def _process_episode(self, env, episode, episodes, train):
        actions, dates, rewards = [], [], []
        state = env.reset()
        max_timesteps = len(env)
        for timestep in range(max_timesteps):
            action = self.select_action(state)
            next_state, reward, done, date = env.step(action)
            if not np.isnan(reward):
                self.replay_buffer.add((state, action, next_state, reward, float(done)))  # Добавление в буфер
                actions.append(action)
                rewards.append(reward)
                dates.append(date)

            state = next_state
            print_progress(timestep, max_timesteps, episode, episodes, reward)

            if done:
                break

        return rewards, actions, dates
