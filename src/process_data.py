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



if __name__ == '__main__':
    main()