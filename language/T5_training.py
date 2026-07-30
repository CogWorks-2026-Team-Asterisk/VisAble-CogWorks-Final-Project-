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

DATA_PATH = "train_dataset.json"

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
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
        )

        target_encoding = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_len,
            truncation=True,
        )

        input_encoding["labels"] = target_encoding["input_ids"]

        return input_encoding

train_dataset = PartToWholeDataset(train_data, tokenizer)
val_dataset = PartToWholeDataset(val_data, tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

training_args = TrainingArguments(
    num_train_epochs = 10, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8, 
    learning_rate = 3e-4,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("t5-part-to-whole-final")
tokenizer.save_pretrained("t5-part-to-whole-final")

test_input = (
    "describe part-to-whole: "
    "Entities: [face, eyes, nose, mouth] "
    "Relations: [face contains eyes, "
    "face contains nose, face contains mouth, "
    "mouth located_below nose]"
)


trained_model = trainer.model
trained_model.eval()


inputs = tokenizer(
    test_input,
    return_tensors="pt",
    max_length=512,
    truncation=True,
)

inputs = {key: value.to(trained_model.device) for key, value in inputs.items()}


with torch.no_grad():
    output = trained_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        num_beams=4,
    )


print(tokenizer.decode(output[0], skip_special_tokens=True))
