import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_preprocessing import BiasDataset, load_data
from tqdm import tqdm
import os

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5

class BiasClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BiasClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)

def main():
    train_df, test_df = load_data('bias_dataset.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    label_cols = ['confirmation', 'overconfidence', 'anchoring']
    
    train_dataset = BiasDataset(
        texts=train_df.input_text.to_numpy(),
        labels=train_df[label_cols].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BiasClassifier(len(label_cols))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bias_classifier.bin")
    print("Model saved to models/bias_classifier.bin")

if __name__ == "__main__":
    main()
