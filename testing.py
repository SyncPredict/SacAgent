import wandb
import json

from src.agent.agent import Agent
from src.agent.buffer import ReplayBuffer
from src.data.data_processor import DataProcessor

# Инициализация W&B
from src.utils.dir import find_directory

processor = DataProcessor('test_data.json')

replay_buffer_size = 105120  # Размер буффера
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_state_dim = 1440
batch_size = 1440
window_size = 288
agent = Agent(window_size, 2, hidden_state_dim, replay_buffer, batch_size)
model_path = find_directory()
print(model_path)
agent.load_model('./models/run_2/episode_108.pth')

results = []  # Список для хранения результатов

def execute():
    for i in range(len(processor) - window_size):
        price = processor.get_price(window_size + i)
        date = processor.get_datetime(window_size + i)
        data = processor.get_episode_data(i, window_size + i)
        stop_loss, take_profit = agent.select_action(data)
        result_price = processor.execute_trading_decision(window_size + i, stop_loss, take_profit)
        profit = result_price - price

        results.append({'date': date, 'profit': profit})  # Добавление результата в список

    # Запись результатов в файл
    with open('results.json', 'w') as f:
        json.dump(results, f)

    # Расчёт метрик
    total_profit = sum([result['profit'] for result in results])
    average_profit = total_profit / len(results)
    positive_results_percentage = (sum([1 for result in results if result['profit'] > 0]) / len(results)) * 100

    # Вывод метрик
    print(f"Сумма всех результатов: {total_profit}")
    print(f"Средний результат: {average_profit}")
    print(f"Процент положительных результатов: {positive_results_percentage}%")




for x in range(10):
    print('\n-----------------------------------\n')
    print('Test №' + str(x) + '\n')
    execute()
    print('\n-----------------------------------\n')
