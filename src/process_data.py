from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Ограничение количества потоков CPU
torch.set_num_threads(4)

def main():

    # Получение абсолютного пути к файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Создание директории data/processed, если она не существует
    processed_dir = os.path.join(current_dir, '../data/processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Загрузка данных
    data_path = os.path.join(current_dir, '../data/raw/train.csv')
    data = pd.read_csv(data_path)

    # Проверка на пропущенные значения в столбце 'Text'
    print(data['Text'].isnull().sum())

    # Замена пропущенных значений на пустые строки
    data['Text'] = data['Text'].fillna('')

    # Извлечение текстов и меток
    texts = data['Text'].values
    labels = data['Sentiment'].values

    # Преобразование меток в числовой формат
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.8, random_state=42)

    # Загрузка токенизатора BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Подготовка данных для BERT
    def prepare_data(texts, labels, tokenizer, max_length=64):
        inputs = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
        return dataset

    train_dataset = prepare_data(X_train, y_train, tokenizer)
    val_dataset = prepare_data(X_val, y_val, tokenizer)

    # Сохранение обработанных данных с использованием абсолютных путей
    train_dataset_path = os.path.join(processed_dir, 'train_dataset.pt')
    val_dataset_path = os.path.join(processed_dir, 'val_dataset.pt')
    torch.save(train_dataset, train_dataset_path)
    torch.save(val_dataset, val_dataset_path)
    print("Data processing completed. Processed data saved to data/processed.")

    # Использование DataLoader с num_workers=0 (для macOS)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

    # Проверка доступности ресурсов
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка предобученной модели BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to(device)

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

    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    epoch = 1
    # Обучение модели
#     model.train()
#     for epoch in range(epoch):
#         model.train()
#         total_loss = 0
#         for i, batch in enumerate(train_loader):
#             optimizer.zero_grad()
#             input_ids, attention_mask, labels = batch
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)
#
#             # Убрали autocast, так как он не поддерживается для MPS
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             if (i + 1) % 50 == 0:  # Печатаем прогресс каждые 50 батчей
#                 avg_loss = total_loss / (i + 1)
#                 # Оценка на валидационной выборке
#                 val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
#                 print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
#
#         scheduler.step()  # Обновление learning rate
#         # Оценка на валидационной выборке после эпохи
#         val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
#         print(f"Epoch {epoch + 1} finished. Average Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
#
#     # Оценка модели после дообучения
#     accuracy_after, f1_after = evaluate_model(model, val_loader, device)
#     print(f"After Fine-tuning - Accuracy: {accuracy_after:.4f}, F1-score: {f1_after:.4f}")

if __name__ == '__main__':
    main()