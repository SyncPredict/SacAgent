import json
import numpy as np
import gc  # Импортируем модуль сборщика мусора
import wandb
import os

from src.utils.dir import find_directory
from src.utils.visualize import print_progress
from src.agent.dqn import DQNAgent  # Предполагается, что dqn.py находится в src/agent

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=3e-4, gamma=0.99, replay_buffer_size=10000, batch_size=64):
        self.agent = DQNAgent(state_dim, action_dim, hidden_dim, lr=lr, gamma=gamma, replay_buffer_size=replay_buffer_size, batch_size=batch_size)

    def execute_episodes(self, env, episodes=20, train=False):
        new_dir = find_directory(True)
        for episode in range(episodes):
            rewards, actions, dates = self._process_episode(env, episode, episodes, train)
            total_rewards = sum(rewards)  # Общая награда за эпизод
            average_reward_per_step = total_rewards / len(rewards) if rewards else 0
            positive_rewards_count = len([r for r in rewards if r > 0])
            win_rate = (positive_rewards_count / len(rewards) * 100) if rewards else 0

            wandb.log({
                'Episode total rewards': total_rewards,
                'Average reward per step': average_reward_per_step,
                'Win rate (%)': win_rate,
                'Rewards distribution': wandb.Histogram(rewards)
            })

            if train:
                self.agent.replay()  # Обновление сети DQN из буфера воспроизведения
                self.agent.save_model(os.path.join(new_dir, f'episode_{episode}.pth'))

            gc.collect()

    def _process_episode(self, env, episode, episodes, train):
        actions, dates, rewards = [], [], []
        state = env.reset()
        max_timesteps = len(env)
        for timestep in range(max_timesteps):
            action = self.agent.select_action(state)
            next_state, reward, done, date = env.step(action)
            if not np.isnan(reward):
                self.agent.add_to_replay_buffer(state, action, reward, next_state, float(done))
                actions.append(action)
                rewards.append(reward)
                dates.append(date)

            state = next_state
            print_progress(timestep, max_timesteps, episode, episodes, reward)

            if done:
                break

        return rewards, actions, dates
