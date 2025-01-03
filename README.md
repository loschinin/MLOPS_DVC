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

### Настройка удаленного хранилища (Yandex)
- Создание бакета в Yandex Object Storage
- Уникальное имя бакета - mlops2train

- Установка и настройка DVC для работы с Yandex Object Storage
- Получение Access Key ID и Secret Access Key

### Синхронизация с удаленным хранилищем.

- dvc remote add -d myremote s3://mlops2train/data

- Параметры для доступа к Yandex Object Storage:
- dvc remote modify myremote endpointurl https://storage.yandexcloud.net
  dvc remote modify myremote access_key_id <your-access-key-id>
  dvc remote modify myremote secret_access_key <your-secret-access-key>

- Проверка конфигурации 
- dvc remote list
  dvc remote modify myremote --list

- Загрузка данных в удаленное хранилище
- dvc push
- Поддягивание данных из удаленного хранилища
- dvc pull

### Создание и запуск пайплайна