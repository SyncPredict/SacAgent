import os


def find_directory(create_new=False, base_directory="./models"):
    number = 0  # Начинаем с run_1
    last_existing_dir = None

    while True:
        directory_path = os.path.join(base_directory, f"run_{number}")
        if not os.path.exists(directory_path):
            if create_new:
                os.makedirs(directory_path)
                return directory_path
            else:
                if last_existing_dir is not None:
                    # Получаем список файлов в последней существующей директории
                    files = os.listdir(last_existing_dir)
                    if files:
                        # Полный путь к последнему файлу
                        last_file = sorted(files)[-1]  # Последний файл по алфавиту
                        last_file_path = os.path.join(last_existing_dir, last_file)
                    else:
                        last_file_path = "Файлы отсутствуют"
                    return last_file_path

        else:
            last_existing_dir = directory_path
            number += 1
