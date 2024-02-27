import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MultiHeadAttention(nn.Module):
    """
    Многоголовочный механизм внимания для улучшенной фокусировки на ключевых аспектах входных данных.
    """

    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert (
                self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
            Прямой проход многоголовочного механизма внимания.

            Параметры:
                x (torch.Tensor): Входной тензор размерности (N, seq_length, hidden_dim),
                                  где N - размер батча, seq_length - длина последовательности, hidden_dim - размер скрытого слоя.

            Возвращает:
                torch.Tensor: Тензор после применения механизма внимания и последующего линейного слоя,
                              размерность (N, seq_length, hidden_dim).
            """
        N, seq_length, _ = x.size()
        query = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        attention = self.softmax(energy)
        out = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(N, seq_length, self.hidden_dim)
        return self.fc_out(out)


class Actor(nn.Module):
    """
    Актерская сеть для SAC агента, использующая LSTM и многоголовочное внимание для обработки временных последовательностей состояний.

    Атрибуты:
        lstm (nn.LSTM): LSTM слой для извлечения признаков из временных последовательностей.
        attention (MultiHeadAttention): Механизм многоголовочного внимания для фокусировки на ключевых аспектах последовательности.
        layer_norm1 (nn.LayerNorm): Слой нормализации для стабилизации выходных данных механизма внимания.
        fc1 (nn.Linear): Полносвязный слой для дополнительной обработки признаков.
        layer_norm2 (nn.LayerNorm): Второй слой нормализации после полносвязного слоя.
        mean_layer (nn.Linear): Слой для вычисления среднего значения распределения действий.
        log_std_layer (nn.Linear): Слой для вычисления логарифма стандартного отклонения распределения действий.
        max_action (float): Максимально возможное значение действия.

    Методы:
        forward(state): Принимает на вход состояние среды и возвращает параметры нормального распределения действий.
        sample(state): Генерирует действие из нормального распределения, заданного параметрами, полученными из forward.
    """

    def __init__(self, state_dim, action_dim, hidden_dim, num_heads):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.max_action = 1.0

    def forward(self, state):
        """
        Прямой проход актерской сети.

        Параметры:
            state (torch.Tensor): Входной тензор состояний размерности (N, seq_length, features),
                                  где N - размер батча, seq_length - длина последовательности, features - количество признаков состояния.

        Возвращает:
            mean (torch.Tensor): Тензор средних значений для нормального распределения действий, размерность (N, seq_length, action_dim).
            log_std (torch.Tensor): Тензор логарифмов стандартных отклонений для нормального распределения действий, размерность (N, seq_length, action_dim).
        """
        lstm_out, _ = self.lstm(state)
        attention_out = self.attention(lstm_out)
        ln_out = self.layer_norm1(attention_out)
        fc_out = F.relu(self.fc1(ln_out))
        ln_out2 = self.layer_norm2(fc_out)

        mean = self.mean_layer(ln_out2)
        log_std = self.log_std_layer(ln_out2)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)  # Получаем предсказания для всей последовательности
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        z = normal_dist.rsample()
        action = torch.sigmoid(z) * self.max_action  # Применяем сигмоиду для получения действия в диапазоне [0, 1]

        # Выбираем последнее действие из последовательности
        processed_action = action[:, -1, :]  # Предполагается, что размерность action [batch_size, seq_length, action_dim]

        log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob[:, -1, :].sum(-1, keepdim=True)  # Выбираем лог-вероятности для последнего действия

        return processed_action, log_prob