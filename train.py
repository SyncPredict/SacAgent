import wandb

from src.agent.agent import Agent
from src.agent.buffer import ReplayBuffer
from src.data.data_processor import DataProcessor
from src.trading_env.env import TradingEnv
from src.utils.dir import find_directory

wandb.init("sac_train_new")

processor = DataProcessor("data.json")

# replay_buffer_size = 105120  # Размер буффера
replay_buffer_size = 400000  # Размер буффера
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_state_dim = 256
batch_size = 32
window_size = 288
input_size = 1  # Размерность входных данных (в данном случае это только цена)
agent = Agent(input_size, 2, hidden_state_dim, replay_buffer, batch_size)
model_path = find_directory()
# agent.load_model(model_path)
# print(f'Loaded: {model_path}')
env = TradingEnv(processor, window_size)
agent.execute_episodes(env, 10000, True)

agent.save_model("sac_model", 3)
