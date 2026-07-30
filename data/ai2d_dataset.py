
import json
import os

from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset


class AI2DDataset(Dataset):

    def __init__(
        self,
        repo_id="abhayzala/AI2D-Caption",
        cache_dir=None,
        transform=None
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.transform = transform

        json_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename="ai2d_caption_gpt4v.json",
            cache_dir=self.cache_dir
        )

        with open(json_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        if not isinstance(self.records, list):
            raise ValueError("The dataset JSON must contain a list.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.get_record(index)
        image_name = self.get_image_name(index)
        image = self.get_image(index)

        return {
            "index": index,
            "image": image,
            "image_name": image_name,
            "caption": self.get_caption(index),
            "entities": self.get_entities(index),
            "entity_labels": self.get_entity_labels(index)
        }

    def _validate_index(self, index):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")

        if index < 0 or index >= len(self.records):
            raise IndexError(
                f"Index must be between 0 and {len(self.records) - 1}."
            )

    def get_record(self, index):
        self._validate_index(index)
        return self.records[index]

    def get_image_name(self, index):
        record = self.get_record(index)
        return os.path.basename(record["image"])

    def get_image(self, index):
        image_path = self.download_image(index)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def get_caption(self, index):
        record = self.get_record(index)
        return str(record.get("caption", "")).strip()

    def get_entities(self, index):
        record = self.get_record(index)
        entities = record.get("entities", {})

        if not isinstance(entities, dict):
            return {}

        return entities

    def get_entity_labels(self, index):
        entities = self.get_entities(index)

        labels = []
        seen = set()

        for entity in entities.values():
            if not isinstance(entity, dict):
                continue

            if entity.get("type") == "relationship":
                continue

            label = str(entity.get("label", "")).strip()

            if not label:
                continue

            if label.lower() == "arrow":
                continue

            normalized_label = label.lower()

            if normalized_label not in seen:
                labels.append(label)
                seen.add(normalized_label)

        return labels

    def download_image(self, index):
        image_name = self.get_image_name(index)

        return hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"ai2d_images/{image_name}",
            cache_dir=self.cache_dir
        )
