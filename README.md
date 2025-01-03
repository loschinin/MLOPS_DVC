## Запуск приложения локально

    python3 -m venv venv  # Создание окружения
    source venv/bin/activate  # Активация в Linux/macOS
    venv\Scripts\activate  # Активация в Windows
    pip install --upgrade pip
    
    pip install -r requirements.txt

## Часть 1: управление данными с DVC
Задача: использовать DVC для управления данными и построения ML-пайплайнов. Настроить удаленное хранилище и запустить пайплайн с использованием CI/CD.

Инициализация DVC
        
    dvc init

Добавление данных в DVC

    dvc add data/raw/train.csv

Коммит изменений

    git add data/raw/train.csv.dvc .gitignore
    git commit -m "Add raw data to DVC"