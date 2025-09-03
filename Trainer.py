import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import time
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import warnings
import re
import unicodedata

warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
EMOTIONS = ['anger', 'fear', 'joy', 'sadness']
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-6

# Model class
class BERTForRegression(nn.Module):
    def __init__(self, bert_model_name='disbert-base-uncased'):
        super(BERTForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regression_layer = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regression_layer(pooled_output)

def clean_tweet(text):
    """Clean tweet text by removing emojis, special characters, and unnecessary spaces."""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove emojis and symbols beyond Basic Multilingual Plane
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove unusual control characters (excluding newlines and tabs)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Training function
# Training function
def train(model, train_loader, val_loader, loss_fn, optimizer, device, save_path, epochs=EPOCHS):
    # Ensure model is on the correct device
    model.to(device)

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"📂 Loaded checkpoint with best val loss: {best_val_loss:.4f}")
    else:
        best_val_loss = 1e10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch[0].to(device)  # Ensure input_ids are on the correct device
            attention_mask = batch[1].to(device)  # Ensure attention_mask is on the correct device
            labels = batch[2].to(device)  # Ensure labels are on the correct device

            if torch.isnan(input_ids).any() or torch.isnan(labels).any():
                print("⚠️ NaNs detected in training batch")
                continue

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()
            outputs = torch.clamp(outputs, 0.0, 1.0)  # Ensure outputs are within the desired range
            loss = loss_fn(outputs, labels)
            if torch.isnan(loss):
                print("⚠️ NaN loss detected during training")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)  # Ensure input_ids are on the correct device
                attention_mask = batch[1].to(device)  # Ensure attention_mask is on the correct device
                labels = batch[2].to(device)  # Ensure labels are on the correct device

                outputs = model(input_ids, attention_mask).squeeze()
                outputs = torch.clamp(outputs, 0.0, 1.0)  # Ensure outputs are within the desired range
                loss = loss_fn(outputs, labels)

                if torch.isnan(loss):
                    print("⚠️ NaN loss in validation")
                    continue

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if not torch.isnan(torch.tensor(avg_val_loss)) and not torch.isinf(torch.tensor(avg_val_loss)):
            if avg_val_loss < best_val_loss and avg_val_loss != 0:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss},
                    save_path)
                print(f"✅ Best model saved with val loss {avg_val_loss:.4f}")
        else:
            print(f"⚠️ Skipping checkpoint due to invalid val loss: {avg_val_loss}")

        print(f"Epoch Time: {time.time() - start_time:.2f}s")
        print('-' * 60)


# Tokenization helper
def tokenize_data(df, tokenizer):
    input_ids, attention_masks, labels = [], [], []
    for tweet, intensity in zip(df['Tweet'], df['Intensity']):
        if pd.isna(tweet) or pd.isna(intensity):
            continue
        try:
            tweet = clean_tweet(str(tweet))
            intensity = float(intensity)
            encoded = tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            labels.append(intensity)
        except Exception as e:
            print(f"⚠️ Skipping problematic entry: {tweet[:30]} | {e}")
            continue

    return TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.tensor(labels, dtype=torch.float32)
    )

# Main loop
for emotion in EMOTIONS:
    print(f"\n🔄 Processing {emotion.upper()}")

    df_path = f'tuned_data\\{emotion}'
    train_df = pd.read_csv(f'{df_path}\\{emotion}_training_data.csv', sep='\t')
    val_df = pd.read_csv(f'{df_path}\\{emotion}_validation_data.csv', sep='\t')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = tokenize_data(train_df, tokenizer)
    val_dataset = tokenize_data(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=BATCH_SIZE)

    model = BERTForRegression()
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model_path = f'models\\bert_regression_{emotion}.pt'
    train(model, train_loader, val_loader, loss_fn, optimizer, device, model_path)

print("\n✅ Training complete for all emotions.")