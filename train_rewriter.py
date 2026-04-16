import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_preprocessing import RewriteDataset, load_data
from tqdm import tqdm
import os

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 3e-4

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)

def main():
    train_df, test_df = load_data('bias_dataset.csv')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Filter only biased samples for rewriting training (optional, but requested pipeline only rewrites if biased)
    # However, for training, we can use the whole set where neutral stays neutral.
    train_dataset = RewriteDataset(
        inputs=train_df.input_text.to_numpy(),
        targets=train_df.rewritten_text.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Starting T5 Training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/t5_rewriter")
    tokenizer.save_pretrained("models/t5_rewriter")
    print("Model saved to models/t5_rewriter")

if __name__ == "__main__":
    main()
