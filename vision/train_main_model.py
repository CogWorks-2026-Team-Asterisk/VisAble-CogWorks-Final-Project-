import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BlipForConditionalGeneration, BlipProcessor, AdamW
from PIL import Image
import json

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

if DATA_DIRECTORY not in sys.path:
    sys.path.append(DATA_DIRECTORY)

from ai2d_dataset import AI2DDataset

class ProcessedAI2DDataset(Dataset):
    def __init__(self, ai2d_dataset, processor, max_length=128):
        self.dataset = ai2d_dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        text = item["caption"]
        
        encoding = self.processor(
            images=image, 
            text=text, 
            padding="max_length", 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True
        )
        
        pixel_values = encoding.pixel_values.squeeze()
        labels = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze() 
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100 

        return {
            "pixel_values": pixel_values, 
            "labels": labels,
            "attention_mask": attention_mask
        }

def create_weighted_loss(vocab_size, entity_token_ids, entity_weight=5.0):
    weights = torch.ones(vocab_size)
    
    for token_id in entity_token_ids:
        if token_id < vocab_size:
            weights[token_id] = entity_weight
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)
    
    return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("vision encoder frozen. only training the text decoder.")

    print("downloading and loading ai2d dataset...")
    raw_dataset = AI2DDataset()

    print("extracting unique entities for loss weights...")
    unique_entities = set()
    for i in range(len(raw_dataset)):
        unique_entities.update(raw_dataset.get_entity_labels(i))

    biological_entity_ids = []
    for word in unique_entities:
        token_id = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(token_id) > 0:
            biological_entity_ids.append(token_id[0])
    
    vocab_size = model.config.text_config.vocab_size
    custom_loss_fn = create_weighted_loss(vocab_size, biological_entity_ids)

    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    raw_train, raw_val = random_split(raw_dataset, [train_size, val_size])

    train_dataset = ProcessedAI2DDataset(raw_train, processor)
    val_dataset = ProcessedAI2DDataset(raw_val, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    epochs = 3
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values, 
                input_ids=labels,
                attention_mask=attention_mask
            )
            logits = outputs.logits 

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = custom_loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            print(f"epoch {epoch+1}   batch {batch_idx+1}/{len(train_dataloader)}   loss: {loss.item():.4f}")

        print(f"--- end of epoch {epoch+1}   average train loss: {total_loss/len(train_dataloader):.4f} ---")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    pixel_values=pixel_values, 
                    input_ids=labels,
                    attention_mask=attention_mask
                )
                logits = outputs.logits 

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = custom_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                val_loss += loss.item()
                
        print(f"--- validation loss for epoch {epoch+1}: {val_loss/len(val_dataloader):.4f} ---")

    model.save_pretrained("./diagram_blip_model")
    processor.save_pretrained("./diagram_blip_model")
    print("model saved to ./diagram_blip_model")

if __name__ == "__main__":
    train()