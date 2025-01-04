## Запуск приложения локально

    python3 -m venv venv  # Создание окружения
    source venv/bin/activate  # Активация в Linux/macOS
    venv\Scripts\activate  # Активация в Windows
    pip install --upgrade pip
    
    pip install -r requirements.txt

## Часть 1: управление данными с DVC
Задача: использовать DVC для управления данными и построения ML-пайплайнов. Настроить удаленное хранилище и запустить пайплайн с использованием CI/CD.

### Добавление данных в DVC

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

#### Синхронизация с удаленным хранилищем.

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
Код в этом файле подготавливает данные для Файнтюнинга: загружает текстовые данные из CSV-файла, предварительно обрабатывает их (заполняет пропуски, кодирует метки, токенизирует тексты с использованием BERT) и разделяет на обучающую и валидационную выборки. Затем он сохраняет обработанные данные в формате .pt для последующего использования в обучении модели.

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

Логирование в CI/CD.


### Итоговый Flow:

Допустим, решили обучить модель на измененных данных - исправили train.csv  
-> проверяем статус dvc status

Первым шагом, запускаем пайплайн repro, чтобы выходные данные тоже изменились и были учтены в DVC

1. dvc repro (пересоздание данных).
-> проверяем статус dvc status

2. git commit (фиксация изменений в Git).

3. dvc push (загрузка данных в удаленное хранилище).

4. git push (отправка кода на гитхаб и запуск CI/CD пайплайна).


В этом задании (Часть 1) использовался DVC для управления данными и построения ML-пайплайнов: данные были добавлены в DVC, настроено удаленное хранилище в Yandex Object Storage, а также создан и запущен пайплайн для обработки данных. Интеграция с CI/CD (GitHub Actions) позволила автоматизировать процесс обработки данных и синхронизации с удаленным хранилищем. Итоговый флоу включает обновление данных, их обработку, фиксацию изменений в Git и автоматический запуск пайплайна через CI/CD.

## Часть 2: Управление экспериментами с MLflow
Задача: Настроить MLflow для управления экспериментами, их сравнения и документирования результатов.

### Настройка MLflow
  
      pip install mlflow

### MLflow-сервер локально

      mlflow ui

http://127.0.0.1:5000/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

### Эксперименты с файнтюнингом BERT

Напишем код, который обучает модель классификации текста на основе DistilBERT, используя предобработанные данные. 

Проведем два эксперимента с разными размерами батчей, логируя параметры и метрики (точность, F1-score) в MLflow для отслеживания результатов. После обучения модель сохраняется в MLflow с примером входных данных и сигнатурой для дальнейшего использования.

1. создадим файл train_bert.py

2. возьмем данные, сохраненные в data/processed и попробуем ими дообучить BERT

3. откроем новый терминал, активируем в нем среду

4. запустим файл командой 

       python src/train_bert.py

### Визуализация

1. создадим файл charts.py
2. извлечем данные из эксперименов
3. запустим код 
4. в директории visualization/batch-size появится файл metrics_comparision.png - это диаграмма, на которой хорошо видны результаты экспериментов

   ![metrics_comparison.png](visualization%2Fbatch-size%2Fmetrics_comparison.png)

### Результаты экспериментов:

#### Первый эксперимент (batch_size=16):
Val Accuracy: 0.2830, 
Val F1: 0.2272,
Duration 4.6min

![batch_16.png](visualization%2Fbatch-size%2Fbatch_16.png)

#### Второй эксперимент (batch_size=8):
Val Accuracy: 0.2754, 
Val F1: 0.1671,
Duration 5.0min

![batch_8.png](visualization%2Fbatch-size%2Fbatch_8.png)

### Сравнение моделей:

Модель с batch_size=16 показала немного лучшие результаты (Duration
4.6min, Val Accuracy: 0.2830, Val F1: 0.2272)

![batch_16_charts.png](visualization%2Fbatch-size%2Fbatch_16_charts.png)

по сравнению с Моделью batch_size=8 (Duration 5.0min, Val Accuracy: 0.2754, Val F1: 0.1671), что может указывать на то, что увеличение размера батча немного улучшает стабильность обучения.


![batch_8_charts.png](visualization%2Fbatch-size%2Fbatch_8_charts.png)


## Общий вывод (по Части 2)

В обоих экспериментах значения Val Accuracy (0.2754 и 0.2830) и Val F1 (0.1671 и 0.2272) низкие, что указывает на то, что модель плохо справляется с задачей классификации на текущих данных и с текущими параметрами.
Это может быть связано с недостаточным количеством данных для обучения (используется только 5% данных) или с тем, что модель не успела достаточно обучиться за одну эпоху.

Низкое количество данных и всего 1 эпоха связано с тем, что отсутствуют вычислительные мощности.
На лучших мощностях файнтюнинг этой модели показывает отличные результаты.



