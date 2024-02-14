import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Критическая сеть для SAC агента, оценивающая Q-значение пар состояние-действие.

    Атрибуты:
        state_dim (int): Размерность входного вектора состояния.
        action_dim (int): Размерность вектора действий.
        hidden_dim (int): Размер скрытых слоев.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Инициализация критика.

        Аргументы:
            state_dim (int): Размерность входного вектора состояния.
            action_dim (int): Размерность вектора действий.
            hidden_dim (int): Размер скрытых слоев.
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU())
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Прямой проход критика.

        Аргументы:
            state (torch.Tensor): Тензор состояния.
            action (torch.Tensor): Тензор действия.

        Возвращает:
            torch.Tensor: Q-значение для пары состояние-действие.
        """
        sa = torch.cat([state, action], 1)
        x = self.layer1(sa)
        x = self.layer2(x)
        q_value = self.layer3(x)
        return q_value

class CriticNetworks:
    """
    Класс-обертка для управления двумя критическими сетями и их целевыми версиями в SAC.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic_1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_dim)

        # Инициализация целевых критиков как копии основных критиков
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Определение оптимизаторов для каждого критика
        self.optimizer_1 = torch.optim.Adam(self.critic_1.parameters())
        self.optimizer_2 = torch.optim.Adam(self.critic_2.parameters())

    def predict(self, states, actions):
        """
        Возвращает Q-значения от обоих критиков для данного состояния и действия.
        """
        return self.critic_1(states, actions), self.critic_2(states, actions)

    def target_predict(self, next_states, next_actions):
        """
        Возвращает Q-значения от обоих целевых критиков для следующего состояния и действия.
        """
        return self.target_critic_1(next_states, next_actions), self.target_critic_2(next_states, next_actions)

    def update_targets(self, tau):
        """
        Обновление целевых критиков с помощью soft update.
        """
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



    def state_dict(self):
        """
        Возвращает состояния обоих критиков и их целевых сетей.
        """
        return {
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'target_critic_1_state_dict': self.target_critic_1.state_dict(),
            'target_critic_2_state_dict': self.target_critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Загружает состояния обоих критиков и их целевых сетей.
        """
        self.critic_1.load_state_dict(state_dict['critic_1_state_dict'])
        self.critic_2.load_state_dict(state_dict['critic_2_state_dict'])
        self.target_critic_1.load_state_dict(state_dict['target_critic_1_state_dict'])
        self.target_critic_2.load_state_dict(state_dict['target_critic_2_state_dict'])
