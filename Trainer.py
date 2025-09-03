import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import transformers as ppb
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import time
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import warnings
from torch.nn import MSELoss
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
emotions = ['anger','fear','joy','sadness']
for i in emotions:
    print(f"Running training for {i}")
    df_path = f'tuned_data\\{i}'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_df = pd.read_csv(f'{df_path}\\{i}_training_data.csv',sep= '\t')
    val_df = pd.read_csv(f'{df_path}\\{i}_validation_data.csv',sep= '\t')
    print(train_df.columns)

    input_ids =[]
    attention_masks = []
    labels =[]
    for tweet,intensity in zip(train_df['Tweet'],train_df['Intensity']):
        encoded=tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length = 128,
            padding ='max_length',
            truncation =True,
            return_attention_mask =True,
            return_tensors= 'pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(float(intensity))

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(input_ids,attention_masks,labels)

    #Loading the encoded data
    batch_size = 32
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size
    )

    input_ids =[]
    attention_masks = []
    labels =[]
    for tweet,intensity in zip(val_df['Tweet'],train_df['Intensity']):
        encoded=tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length = 128,
            padding ='max_length',
            truncation =True,
            return_attention_mask =True,
            return_tensors= 'pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(float(intensity))
        
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.float32)
    val_dataset = TensorDataset(input_ids,attention_masks,labels)

    val_dataloader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=batch_size
    )


    class DistilBERTForRegression(nn.Module):
        def __init__(self, model_name='distilbert-base-uncased'):
            super(DistilBERTForRegression, self).__init__()
            self.bert = DistilBertModel.from_pretrained(model_name)
            self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state
            pooled_output = hidden_state[:, 0] 
            return self.sigmoid(self.regressor(pooled_output))
        
    model = DistilBERTForRegression()
    loss_fn = MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, num_epochs=10):
        model = model.to(device)
        best_val_loss = float('inf') 
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            
            # Loop over batches
            for batch in train_dataloader:
                # Get input data from the batch
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids, attention_mask)

                # Calculate the loss
                loss = loss_fn(outputs.squeeze(), labels)  # outputs are (batch_size, 1), labels are (batch_size)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            # Calculate average loss for this epoch
            avg_train_loss = running_loss / len(train_dataloader)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            
            # Evaluate the model on the validation set
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[2].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs.squeeze(), labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'models/bert_regression_{i}.pt')
                print(f"✅ Saved new best model with val loss: {avg_val_loss:.4f}")
            
            epoch_time = time.time() - start_time
            print(f"Epoch Time: {epoch_time:.2f} seconds")
            print("-" * 50)

    # Use the trained model
    train(model, dataloader, val_dataloader, loss_fn, optimizer, device)