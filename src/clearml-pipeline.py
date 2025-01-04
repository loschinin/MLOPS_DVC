from clearml import PipelineDecorator, Task
import subprocess
import time

@PipelineDecorator.component(cache=False, execution_queue="default")
def dvc_pull():
    try:
        print("Starting dvc_pull...")
        subprocess.run(["dvc", "pull"], check=True)
        print("dvc_pull completed successfully.")
        return Task.current_task().id  # Возвращаем task_id для использования в process_data
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_pull: {e}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["dvc_pull"])
def process_data(dvc_pull_task_id):
    try:
        print("Starting process_data...")

        # Ожидание завершения dvc_pull
        dvc_pull_task = Task.get_task(task_id=dvc_pull_task_id)
        while not dvc_pull_task.completed:
            time.sleep(5)  # Ожидание 5 секунд перед повторной проверкой

        # Запуск обработки данных
        subprocess.run(["python", "src/process_data.py"], check=True)
        print("process_data completed successfully.")
        return Task.current_task().id  # Возвращаем task_id для использования в train_model
    except subprocess.CalledProcessError as e:
        print(f"Error in process_data: {e}")
        raise

@PipelineDecorator.component(execution_queue="default", parents=["process_data"])
def train_model(process_data_task_id, batch_size):
    try:
        print(f"Starting train_model with batch_size={batch_size}...")

        # Ожидание завершения process_data
        process_data_task = Task.get_task(task_id=process_data_task_id)
        while not process_data_task.completed:
            time.sleep(5)  # Ожидание 5 секунд перед повторной проверкой

        # Запуск обучения модели с указанным batch_size
        subprocess.run(["python", "src/train_bert.py", "--batch-size", str(batch_size)], check=True)
        print(f"train_model with batch_size={batch_size} completed successfully.")
        return Task.current_task().id  # Возвращаем task_id для использования в dvc_repro
    except subprocess.CalledProcessError as e:
        print(f"Error in train_model: {e}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["train_model"])
def dvc_repro(train_model_task_id):
    try:
        print("Starting dvc_repro...")

        # Ожидание завершения train_model
        train_model_task = Task.get_task(task_id=train_model_task_id)
        while not train_model_task.completed:
            time.sleep(5)  # Ожидание 5 секунд перед повторной проверкой

        # Запуск dvc repro для обновления данных
        subprocess.run(["dvc", "repro"], check=True)
        print("dvc_repro completed successfully.")
        return Task.current_task().id  # Возвращаем task_id для использования в dvc_push
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_repro: {e}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["dvc_repro"])
def dvc_push(dvc_repro_task_id):
    try:
        print("Starting dvc_push...")

        # Ожидание завершения dvc_repro
        dvc_repro_task = Task.get_task(task_id=dvc_repro_task_id)
        while not dvc_repro_task.completed:
            time.sleep(5)  # Ожидание 5 секунд перед повторной проверкой

        # Загрузка данных в удаленное хранилище DVC
        subprocess.run(["dvc", "push"], check=True)
        print("dvc_push completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_push: {e}")
        raise

@PipelineDecorator.pipeline(
    name='text_classification_pipeline',
    project='Text Classification',
    version='0.1'
)
def text_classification_pipeline_logic():
    dvc_pull_task_id = dvc_pull()
    process_data_task_id = process_data(dvc_pull_task_id)

    # Запуск моделей с разными batch_size последовательно
    train_model_task_id_8 = train_model(process_data_task_id, batch_size=8)
    train_model_task_id_16 = train_model(train_model_task_id_8, batch_size=16)  # Зависит от завершения первой модели

    # Обновление DVC и загрузка данных
    dvc_repro_task_id = dvc_repro(train_model_task_id_16)
    dvc_push(dvc_repro_task_id)

if __name__ == '__main__':
    # Запуск пайплайна локально (для отладки)
    PipelineDecorator.run_locally()
    text_classification_pipeline_logic()