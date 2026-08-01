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
# I'm running this on Colab; kept here for reference.

DATA_PATH = "training_data.json"

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
    def __init__(self, data, tokenizer, max_input_len=512, max_target_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

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
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


train_dataset = PartToWholeDataset(train_data, tokenizer)
val_dataset = PartToWholeDataset(val_data, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
trainer.train()

model.save_pretrained("t5-part-to-whole-final")
tokenizer.save_pretrained("t5-part-to-whole-final")

test_input = "describe part-to-whole: Entities: [...] Relations: [...]"
output = model.generate(**tokenizer(test_input, return_tensors="pt"))
print(tokenizer.decode(output[0], skip_special_tokens=True))
