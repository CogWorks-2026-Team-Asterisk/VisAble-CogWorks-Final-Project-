import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor, AdamW
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

if DATA_DIRECTORY not in sys.path:
    sys.path.append(DATA_DIRECTORY)

from ai2d_dataset import AI2DDataset


class PartToWholeDataset(Dataset):
    def __init__(self, json_file, processor, max_length=256):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["image_path"]
        text = item["part_to_whole_text"]

        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        pixel_values = encoding.pixel_values.squeeze()
        labels = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_weighted_loss(vocab_size, entity_token_ids, entity_weight=5.0):
    weights = torch.ones(vocab_size)

    for token_id in entity_token_ids:
        if token_id < vocab_size:
            weights[token_id] = entity_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)

    return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)


def get_biological_entity_ids(processor):
    print("extracting unique entities from ai2ddataset to create loss weights")

    raw_dataset = AI2DDataset()
    unique_entities = set()

    for i in range(len(raw_dataset)):
        unique_entities.update(raw_dataset.get_entity_labels(i))

    entity_ids = []
    for word in unique_entities:
        token_ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 0:
            entity_ids.append(token_ids[0])

    return entity_ids


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    for param in model.vision_model.parameters():
        param.requires_grad = False

    print("vision encoder frozen only training the text decoder")

    entity_ids = get_biological_entity_ids(processor)
    vocab_size = model.config.text_config.vocab_size
    custom_loss_fn = create_weighted_loss(vocab_size, entity_ids, entity_weight=5.0)

    dataset_path = os.path.join(DATA_DIRECTORY, "ptw_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cannot find {dataset_path}. Run modifying_data.py first!")

    train_dataset = PartToWholeDataset(json_file=dataset_path, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
    )

    epochs = 3
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=labels,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = custom_loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"epoch {epoch + 1} batch {batch_idx} {len(train_dataloader)} loss {loss.item():.4f}"
                )

        print(
            f"end of epoch {epoch + 1} average loss {total_loss / len(train_dataloader):.4f}"
        )

    output_dir = os.path.join(PROJECT_ROOT, "vision", "diagram_blip_model")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"model saved to {output_dir}")


if __name__ == "__main__":
    train_model()
