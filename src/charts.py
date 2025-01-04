import mlflow
import matplotlib.pyplot as plt
import pandas as pd

# tracking_uri на адрес MLflow-сервера
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Извлечение данных о запущенных экспериментах
runs = mlflow.search_runs(experiment_ids=["0"])  # ID эксперимента (например, "0")
print("Available columns in runs data:", runs.columns)  # Вывод всех доступных колонок

# Проверка наличия нужных колонок
if all(col in runs.columns for col in ["params.batch_size", "metrics.val_accuracy", "metrics.val_f1"]):
    # Фильтрация данных: удаляем строки, где params.batch_size равен None
    runs = runs[runs["params.batch_size"].notna()]

    # Преобразование данных
    runs["params.batch_size"] = runs["params.batch_size"].astype(int)

    # Преобразование метрик в числа
    runs["metrics.val_accuracy"] = pd.to_numeric(runs["metrics.val_accuracy"], errors="coerce")
    runs["metrics.val_f1"] = pd.to_numeric(runs["metrics.val_f1"], errors="coerce")

    # Удаление строк с NaN в метриках (если такие есть)
    runs = runs.dropna(subset=["metrics.val_accuracy", "metrics.val_f1"])

    # Группировка и вычисление средних значений
    runs = runs.groupby("params.batch_size").mean(numeric_only=True).reset_index()

    # Вывод данных для проверки
    print(runs)

    # Построение столбчатых диаграмм
    plt.figure(figsize=(10, 5))

    # График для Accuracy
    plt.subplot(1, 2, 1)
    plt.bar(runs["params.batch_size"].astype(str), runs["metrics.val_accuracy"], color='blue')
    plt.xlabel("Batch Size")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy by Batch Size")

    # График для F1-score
    plt.subplot(1, 2, 2)
    plt.bar(runs["params.batch_size"].astype(str), runs["metrics.val_f1"], color='green')
    plt.xlabel("Batch Size")
    plt.ylabel("Validation F1 Score")
    plt.title("F1 Score by Batch Size")

    # Сохранение графиков
    plt.tight_layout()
    plt.savefig("batch_size/visualization/metrics_comparison.png")
    plt.show()
else:
    print("Required columns are missing in the runs data.")