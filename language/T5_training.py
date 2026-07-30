import json
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
#I'm running this on collab, I just put it here

DATA_PATH = "train_dataset_partial.json"

print("Loading training data...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

split_index = int(len(data) * 0.8)
train_data = data[:split_index]
val_data = data[split_index:]

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

class PartToWholeDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len = 512, max_target_len = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = 512
        self.max_target_len = 512
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
            
        item = self.data[idx]
        input_text = item["input"]   
        target_text = item["output"]
        

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"   # return as PyTorch tensors
        )

        # Tokenize the target (y)
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

train_dataset = PartToWholeDataset(train_data, tokenizer)
val_dataset = PartToWholeDataset(val_data, tokenizer)

training_args = TrainingArguments(
    epochs = 10, batch_size = 8, learning_rate = 3e-4,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    fp16 = torch.cuda.is_available()
)

trainer = Trainer(model, training_args, train_dataset, val_dataset)
trainer.train()

model.save_pretrained("t5-part-to-whole-final")
tokenizer.save_pretrained("t5-part-to-whole-final")

test_input = "describe part-to-whole: Entities: [...] Relations: [...]"
output = model.generate(tokenizer(test_input))
print(tokenizer.decode(output))

