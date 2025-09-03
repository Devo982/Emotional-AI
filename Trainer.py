import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import time
from torch.optim import AdamW
from torch.nn import MSELoss
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

emotions = [ 'anger']

for i in emotions:
    print(f"\n🔁 Running training for {i}")
    df_path = f'tuned_data\\{i}'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_df = pd.read_csv(f'{df_path}\\{i}_training_data.csv', sep='\t')
    val_df = pd.read_csv(f'{df_path}\\{i}_validation_data.csv', sep='\t')

    def encode_data(df):
        input_ids, attention_masks, labels = [], [], []
        for tweet, intensity in zip(df['Tweet'], df['Intensity']):
            encoded = tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze())
            attention_masks.append(encoded['attention_mask'].squeeze())
            labels.append(float(intensity))
        return (
            torch.stack(input_ids),
            torch.stack(attention_masks),
            torch.tensor(labels, dtype=torch.float32)
        )

    train_input_ids, train_attention_masks, train_labels = encode_data(train_df)
    val_input_ids, val_attention_masks, val_labels = encode_data(val_df)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    batch_size = 32
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)

    class DistilBERTForRegression(nn.Module):
        def __init__(self):
            super(DistilBERTForRegression, self).__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            return self.sigmoid(self.regressor(pooled_output))

    model = DistilBERTForRegression().to(device)
    loss_fn = MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    def has_nan_or_inf(tensor):
        return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()

    def train(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs=10):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            nan_batch_count = 0

            for step, batch in enumerate(train_loader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                # Check inputs for NaNs or Infs
                if any([
                    has_nan_or_inf(input_ids),
                    has_nan_or_inf(attention_mask),
                    has_nan_or_inf(labels)
                ]):
                    print(f"🚨 Skipping batch {step} due to NaN/Inf in inputs/labels")
                    nan_batch_count += 1
                    continue

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)

                if has_nan_or_inf(outputs):
                    print(f"🚨 Skipping batch {step} due to NaN/Inf in outputs")
                    nan_batch_count += 1
                    continue

                loss = loss_fn(outputs.squeeze(), labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ NaN loss encountered at batch {step}, skipping update.")
                    nan_batch_count += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # removing out of bound values
                optimizer.step()

                if any([has_nan_or_inf(p.grad) for p in model.parameters() if p.grad is not None]):
                    print(f"💥 Gradient explosion detected at batch {step}, skipping optimizer step")
                    nan_batch_count += 1
                    continue

                optimizer.step()
                running_loss += loss.item()

                if step % 10 == 0:
                    print(f"[Batch {step}] Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / (len(train_loader) - nan_batch_count + 1e-6)
            print(f"✅ Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Skipped Batches: {nan_batch_count}")

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids = val_batch[0].to(device)
                    attention_mask = val_batch[1].to(device)
                    labels = val_batch[2].to(device)
                    outputs = model(input_ids, attention_mask)

                    if has_nan_or_inf(outputs):
                        print("🚫 Skipping val batch due to NaNs")
                        break

                    loss = loss_fn(outputs.squeeze(), labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"🧪 Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss and avg_val_loss!=0:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'models/bert_regression_{i}.pt')
                print(f"💾 Saved best model for {i} with val loss {avg_val_loss:.4f}")

            print(f"⏱️ Epoch Time: {time.time() - start_time:.2f}s\n" + "-" * 50)

    train(model, train_loader, val_loader, loss_fn, optimizer, device)