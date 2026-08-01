import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.ai2d_dataset import AI2DDataset

MODEL_NAME = "Salesforce/blip-image-captioning-base"


class PartToWholeDataset(Dataset):
    def __init__(self, json_file, processor, max_length=256):
        with open(json_file, "r") as f:
            records = json.load(f)

        self.data = [
            item
            for item in records
            if str(item.get("part_to_whole_text", "")).strip()
            and os.path.exists(item.get("image_path", ""))
        ]

        skipped = len(records) - len(self.data)
        if skipped:
            print(f"skipped {skipped} bad records")

        if not self.data:
            raise ValueError(f"No usable records found in {json_file}")

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

        pixel_values = encoding.pixel_values.squeeze(0)
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # mask padding for the loss
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_weighted_loss(vocab_size, entity_token_ids, entity_weight=5.0, device=None):
    weights = torch.ones(vocab_size)

    for token_id in entity_token_ids:
        if token_id < vocab_size:
            weights[token_id] = entity_weight

    if device is not None:
        weights = weights.to(device)

    return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)


def get_biological_entity_ids(processor, snapshot_dir=None):
    print("extracting entities for loss weights")

    raw_dataset = AI2DDataset(snapshot_dir=snapshot_dir)
    unique_entities = set()

    for i in range(len(raw_dataset)):
        unique_entities.update(raw_dataset.get_entity_labels(i))

    entity_ids = []
    for word in unique_entities:
        token_ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 0:
            entity_ids.append(token_ids[0])

    print(f"found {len(unique_entities)} entity labels")

    return entity_ids


def train_model(
    dataset_path=None,
    output_dir=None,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    entity_weight=5.0,
    num_workers=2,
    snapshot_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    dataset_path = dataset_path or os.path.join(DATA_DIRECTORY, "ptw_dataset.json")
    output_dir = output_dir or os.path.join(PROJECT_ROOT, "vision", "diagram_blip_model")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Cannot find {dataset_path}. Run modifying_data.py first!"
        )

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    for param in model.vision_model.parameters():
        param.requires_grad = False

    print("vision encoder frozen")

    entity_ids = get_biological_entity_ids(processor, snapshot_dir=snapshot_dir)
    vocab_size = model.text_decoder.cls.predictions.decoder.out_features
    custom_loss_fn = create_weighted_loss(
        vocab_size, entity_ids, entity_weight=entity_weight, device=device
    )

    train_dataset = PartToWholeDataset(json_file=dataset_path, processor=processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"training on {len(train_dataset)} examples")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = custom_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"epoch {epoch + 1} batch {batch_idx}/{len(train_dataloader)} loss {loss.item():.4f}",
                    flush=True,
                )

        print(
            f"epoch {epoch + 1} average loss {total_loss / len(train_dataloader):.4f}"
        )

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"saved epoch {epoch + 1}")

    print(f"model saved to {output_dir}")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BLIP on part-to-whole diagram captions."
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--entity-weight", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Local folder holding the downloaded AI2D-Caption repo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entity_weight=args.entity_weight,
        num_workers=args.num_workers,
        snapshot_dir=args.snapshot_dir,
    )
