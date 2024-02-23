import os

def find_directory(create_new=False, base_directory="./models"):
    number = 1  # Начинаем с run_1
    last_existing_dir = None

    while True:
        directory_path = os.path.join(base_directory, f"run_{number}")
        if not os.path.exists(directory_path):
            if create_new:
                os.makedirs(directory_path)
                return directory_path
            elif last_existing_dir:  # Если last_existing_dir не None и директории нет
                files = os.listdir(last_existing_dir)
                if files:
                    last_file = sorted(files)[-1]  # Последний файл по алфавиту
                    last_file_path = os.path.join(last_existing_dir, last_file)
                    return last_file_path
                else:
                    return "Файлы отсутствуют"
            else:
                return "Директория не найдена и не была создана"
        else:
            last_existing_dir = directory_path
            number += 1
