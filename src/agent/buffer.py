import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        # Явно удаляем каждый элемент в buffer
        for i in range(len(self.buffer)):
            self.buffer[i] = None  # Помогает в освобождении памяти, если есть большие объекты
        # Теперь, когда все элементы установлены в None, можно безопасно очистить buffer
        self.buffer.clear()  # Очищаем список
        self.position = 0

