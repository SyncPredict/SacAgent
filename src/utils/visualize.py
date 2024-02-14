def print_progress(timestep, timesteps, episode, episodes, reward):
    progress = (timestep + 1) / timesteps
    bar_length = 40
    bar = '#' * int(progress * bar_length) + '-' * (bar_length - int(progress * bar_length))
    string = f'\r[{bar}] Episode {episode + 1}/{episodes}, Step {timestep + 1}/{timesteps} ({progress * 100:.2f}%) Reward: {reward:.6f}'
    end = '\n' if timestep + 1 == timesteps else ''
    print(string, end=end)
