import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticNetworks
from ..utils.dir import find_directory


def clear_memory():
    torch.cuda.empty_cache()


class SACAgent:
    """
    Агент, использующий алгоритм Soft Actor-Critic (SAC) для обучения политики в заданной среде.

    Атрибуты:
        actor (Actor): Нейронная сеть, определяющая политику действий агента.
        critic_networks (CriticNetworks): Две критические сети (Q-сети) и две целевые критические сети для оценки действий.
        actor_optimizer (torch.optim.AdamW): Оптимизатор для обновления весов сети актора.
        gamma (float): Коэффициент дисконтирования будущих наград.
        tau (float): Коэффициент для мягкого обновления весов целевых сетей.
        alpha (float): Коэффициент, определяющий важность энтропийного бонуса.
        device (torch.device): Устройство, на котором выполняются вычисления (CPU или GPU).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        num_heads=4,
    ):
        """
        Инициализация агента SAC с заданными параметрами сети и обучения.

        Параметры:
            state_dim (int): Размерность пространства состояний среды.
            action_dim (int): Размерность пространства действий агента.
            hidden_dim (int): Размер скрытых слоёв нейронной сети.
            lr (float): Скорость обучения для оптимизатора.
            gamma (float): Коэффициент дисконтирования будущих наград.
            tau (float): Коэффициент мягкого обновления целевых сетей.
            alpha (float): Коэффициент, определяющий важность энтропийного бонуса.
        """
        self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim, num_heads, self.device)
        self.critic_networks = CriticNetworks(state_dim, action_dim, hidden_dim)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state):
        """
        Выбор действия агентом для заданного состояния.

        Параметры:
            state (np.ndarray): Входной массив состояний размерности (seq_length, ),
                                где seq_length - длина последовательности, features - количество признаков состояния (здесь 1 так как используем только цену).
        Возвращает:
            action (np.ndarray): Выбранное действие в формате numpy массива.
                                 Формат: (action_dim,)

        Пример:
            state = [0.1, 0.2, 0.3]
            action = agent.select_action(state)  # Например, np.array([0.5, -0.1])
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, _ = self.actor.sample(state)  # action имеет форму [1, action_dim]
            action = action.squeeze(0)  # Убираем измерение batch, получаем [action_dim]
        return action.cpu().numpy()

    def update_parameters(self, replay_buffer, batch_size):
        """
        Обновление параметров политики и Q-функций на основе мини-батча из буфера воспроизведения.

        Параметры:
            replay_buffer (ReplayBuffer): Буфер воспроизведения для выборки опыта.
            batch_size (int): Размер мини-батча для обновления.

        Формат входных данных:
            - states: torch.Tensor, формат (batch_size, state_dim)
            - actions: torch.Tensor, формат (batch_size, action_dim)
            - rewards: torch.Tensor, формат (batch_size, 1)
            - next_states: torch.Tensor, формат (batch_size, state_dim)
            - dones: torch.Tensor, формат (batch_size, 1)
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(-1)

        # Обновление Critic
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q_target_next_1, q_target_next_2 = self.critic_networks.target_predict(
                next_states, next_actions
            )
            min_q_target_next = (
                torch.min(q_target_next_1, q_target_next_2) - self.alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * min_q_target_next

        current_q1, current_q2 = self.critic_networks.predict(states, actions)

        # Вычисление потерь отдельно для каждого критика
        critic_loss_1 = F.mse_loss(current_q1, q_target)
        critic_loss_2 = F.mse_loss(current_q2, q_target)

        self.critic_networks.optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_networks.optimizer_1.step()

        # Обновление для второго критика
        self.critic_networks.optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_networks.optimizer_2.step()

        # Обновление Actor
        new_actions, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic_networks.predict(states, new_actions)
        q_new_actions = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update целевых сетей
        self.critic_networks.update_targets(self.tau)

    def save_model(self, new_dir, name):
        """
        Сохраняет модель агента в указанной директории.

        Параметры:
            new_dir (str): Путь к директории для сохранения модели.
            name (str): Имя файла для сохранения состояния модели.
        """
        # Обновляем путь для сохранения, используя имя файла
        save_path = os.path.join(new_dir, f"{name}.pth")

        # Сохраняем состояние модели
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_networks_state_dict": self.critic_networks.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_networks_optimizer_1_state_dict": self.critic_networks.optimizer_1.state_dict(),
                "critic_networks_optimizer_2_state_dict": self.critic_networks.optimizer_2.state_dict(),
            },
            save_path,
        )

    def load_model(self, path, device="cpu"):
        """
        Загружает модель агента из файла.

        Параметры:
            path (str): Путь к файлу с сохраненной моделью.
            device (str): Устройство, на которое будет загружена модель ('cpu' или 'cuda').
        """
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_networks.load_state_dict(checkpoint["critic_networks_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_networks.optimizer_1.load_state_dict(
            checkpoint["critic_networks_optimizer_1_state_dict"]
        )
        self.critic_networks.optimizer_2.load_state_dict(
            checkpoint["critic_networks_optimizer_2_state_dict"]
        )
