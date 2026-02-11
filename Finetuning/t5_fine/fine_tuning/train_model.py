import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW

# -------------------- Config -------------------- #
DATASET_PATH = "/mnt/data/augmented_cases.json"   # new dataset with 2000 samples
LOAD_DIR = "../saved_model_v4"                       # previously saved model
SAVE_DIR = "../saved_model_v5"                    # save new fine-tuned version
BATCH_SIZE = 4
EPOCHS = 3                                        # fewer epochs for refinement
LR = 5e-5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Dataset Class -------------------- #
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['input']
        target = item['output']

        source_enc = self.tokenizer(
            source,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': source_enc['input_ids'].squeeze(),
            'attention_mask': source_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze()
        }

# -------------------- Load Dataset -------------------- #
with open(DATASET_PATH, 'r') as f:
    dataset_json = json.load(f)

# Use 90/10 split for training/validation since dataset is small
train_size = int(0.9 * len(dataset_json))
train_data = dataset_json[:train_size]
val_data = dataset_json[train_size:]

# Load tokenizer & model from previously saved checkpoint
tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(LOAD_DIR).to(DEVICE)

train_ds = CodeDataset(train_data, tokenizer, MAX_LEN)
val_ds = CodeDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LR)

# -------------------- Training Loop -------------------- #
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

# -------------------- Save Updated Model -------------------- #
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"[SUCCESS] Fine-tuned model saved to {SAVE_DIR}")
