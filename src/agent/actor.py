import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Actor(nn.Module):
    """
    Актерская сеть для SAC агента, использующая нормализацию слоев и параметризованные действия для
    предсказания уровней тейк-профит и стоп-лосс.

    Аргументы:
        state_dim (int): Размерность входного вектора состояния.
        action_dim (int): Размерность выходного вектора действия.
        hidden_dim (int): Размер скрытых слоёв.
        max_action (float): Максимально возможное значение действия.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Инициализация актера с нормализацией слоев и параметризованными действиями.

        Аргументы:
            state_dim (int): Размерность входного вектора состояния.
            action_dim (int): Размерность выходного вектора действия.
            hidden_dim (int): Размер скрытых слоёв.
        """
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.max_action = 1.0

    def forward(self, state):
        """
        Вычисляет среднее и логарифм стандартного отклонения действий для заданного состояния.

        Аргументы:
            state (torch.Tensor): Входное состояние.

        Возвращает:
            mean (torch.Tensor): Среднее действий.
            log_std (torch.Tensor): Логарифм стандартного отклонения действий.
        """
        net_out = self.net(state)
        mean = self.mean_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Ограничиваем для стабильности
        return mean, log_std

    def sample(self, state):
        """
        Генерирует действие и его логарифм вероятности для заданного состояния, используя репараметризацию.

        Аргументы:
            state (torch.Tensor): Входное состояние.

        Возвращает:
            action (torch.Tensor): Сгенерированное действие.
            log_prob (torch.Tensor): Логарифм вероятности действия.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        z = normal_dist.rsample()  # Репараметризация для градиентов
        action = torch.sigmoid(z)
        action = self.max_action * action  # Масштабируем действие

        # Расчет логарифма вероятности с учетом преобразования tanh
        log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

