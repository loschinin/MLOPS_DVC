import mlflow
import mlflow.pytorch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.metrics import accuracy_score, f1_score
import os
import numpy as np

# Ограничение количества потоков CPU
torch.set_num_threads(2)

def main():
    # tracking_uri на адрес MLflow-сервера
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Получение абсолютного пути к файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Загрузка предобработанных данных
    processed_dir = os.path.join(current_dir, '../data/processed')
    train_dataset_path = os.path.join(processed_dir, 'train_dataset.pt')
    val_dataset_path = os.path.join(processed_dir, 'val_dataset.pt')

    train_dataset = torch.load(train_dataset_path, weights_only=False)
    val_dataset = torch.load(val_dataset_path, weights_only=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Проверка уникальных меток
    train_labels = train_dataset.tensors[2]
    val_labels = val_dataset.tensors[2]
    print("Unique train labels:", torch.unique(train_labels))
    print("Unique validation labels:", torch.unique(val_labels))

    # Используем CPU вместо MPS
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Функция оценки модели
    def evaluate_model(model, data_loader, device):
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, f1

    # Параметры экспериментов
    experiments = [
        {"batch_size": 8},  # Эксперимент 1
        {"batch_size": 16},  # Эксперимент 2
    ]

    # Общие параметры
    learning_rate = 5e-5
    epochs = 1  # Одна эпоха
    data_subset = 0.05  # Фиксированное подмножество данных (5%)

    # Создание подмножества данных (5%)
    subset_size = int(len(train_dataset) * data_subset)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)

    for exp_id, exp_params in enumerate(experiments):
        with mlflow.start_run():
            print(f"Running Experiment {exp_id + 1} with params: {exp_params}")

            # Логирование параметров
            mlflow.log_param("batch_size", exp_params["batch_size"])
            mlflow.log_param("data_subset", data_subset)
            mlflow.log_param("learning_rate", learning_rate)

            # Использование DataLoader
            train_loader = DataLoader(train_subset, batch_size=exp_params["batch_size"], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=exp_params["batch_size"], num_workers=0)

            # Загрузка предобученной модели DistilBERT с 6 классами
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
            model.to(device)

            # Оптимизатор
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

            # Обучение модели
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for i, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Оценка на валидационной выборке
                val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
                print(f"Experiment {exp_id + 1}, Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

                # Логирование метрик
                mlflow.log_metric("loss", total_loss / len(train_loader), step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

            # Пример входных данных
            input_example = {
                "input_ids": torch.tensor([[101, 2054, 2003, 1996, 2627, 102]]).to(device).cpu().numpy().tolist(),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]).to(device).cpu().numpy().tolist()
            }

            # Автоматическое определение сигнатуры
            signature = mlflow.models.infer_signature(
                input_example,
                model(
                    torch.tensor(input_example["input_ids"]).to(device),
                    torch.tensor(input_example["attention_mask"]).to(device)
                ).logits.detach().cpu().numpy().tolist()  # Используем .detach() перед .numpy()
            )

            # Сохранение модели
            mlflow.pytorch.log_model(model, "model", signature=signature, input_example=input_example)

if __name__ == '__main__':
    main()