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

      dvc remote add -d myremote s3://mlops2train/data

- Параметры для доступа к Yandex Object Storage:
  
      dvc remote modify myremote endpointurl https://storage.yandexcloud.net
      dvc remote modify myremote access_key_id <your-access-key-id>
      dvc remote modify myremote secret_access_key <your-secret-access-key>

- Проверка конфигурации 

      dvc remote list
      dvc remote modify myremote --list

- Загрузка данных в удаленное хранилище

      dvc push

- Поддягивание данных из удаленного хранилища

      dvc pull


### Создание и запуск пайплайна

#### Создадим папку src и в ней файл process_data.py
Код в этом файле подготавливает данные и настраивает модель BERT для задачи классификации текста: он загружает данные из CSV-файла, проверяет и заполняет пропущенные значения, преобразует тексты и метки в числовой формат, разделяет данные на обучающую и валидационную выборки, токенизирует тексты с использованием токенизатора BERT, создает наборы данных и сохраняет их в файлы. Затем данные загружаются в DataLoader для удобной работы с батчами, определяется устройство для вычислений (CPU или MPS для macOS), и загружается предобученная модель BERT для классификации текста. Код подготавливает всё необходимое для последующего обучения модели.

#### Создание файла dvc.yaml

    stages:
      process_data:
        cmd: python src/process_data.py
        deps:
          - data/raw/train.csv
        outs:
          - data/processed/train_dataset.pt
          - data/processed/val_dataset.pt
          

#### Запуск пайплайна

    dvc repro

DVC автоматически проверил изменения в зависимостях и выполнил только те этапы, которые требуют обновления. В директории data появилась поддиректория processed с файлами
train_dataset.pt и val_dataset.pt:

Это файлы содержат обработанные данные, которые были подготовлены для обучения модели

### Коммит и dvc push

    dvc push

3 files pushed - 3 файла были загружены в удаленное хранилище DVC - данные, которые обработали и сохранили в файлах train_dataset.pt и val_dataset.pt, теперь синхронизированы с удаленным хранилищем. И другие участники проекта могут загрузить их с помощью команды dvc pull                         

### Интеграция DVC в CI/CD:
Настроим пайплайн в CI/CD, который будет автоматически запускать DVC-процесс.

#### Создание файла CI/CD (GitHub Actions)

.github/workflows/dvc-pipeline.yml

#### Настройка секретов в GitHub

Settings → Secrets and variables → Actions.

DVC_REMOTE_ACCESS_KEY_ID: Access Key ID для удаленного хранилища.

DVC_REMOTE_SECRET_ACCESS_KEY: Secret Access Key для удаленного хранилища.

