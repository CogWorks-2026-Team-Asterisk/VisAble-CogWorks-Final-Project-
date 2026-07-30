
import os
import sys

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

if DATA_DIRECTORY not in sys.path:
    sys.path.append(DATA_DIRECTORY)

from ai2d_dataset import AI2DDataset


MODEL_NAME = "Salesforce/blip-image-captioning-base"


class ImageToModel:

    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.processor = BlipProcessor.from_pretrained(
            self.model_name
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def prepare_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("The image must be a PIL Image.")

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    def pass_image_to_model(self, image):
        inputs = self.prepare_image(image)

        with torch.no_grad():
            outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True
            )

        return outputs

    def process_dataset_example(self, dataset, index):
        if index < 0 or index >= len(dataset):
            raise IndexError("Dataset index is out of range.")

        example = dataset[index]
        outputs = self.pass_image_to_model(example["image"])

        return {
            "index": index,
            "image": example["image"],
            "image_name": example["image_name"],
            "vision_output": outputs
        }


def test_with_ai2d_example(example_index=0):
    dataset = AI2DDataset()
    image_to_model = ImageToModel()

    return image_to_model.process_dataset_example(
        dataset=dataset,
        index=example_index
    )


if __name__ == "__main__":
    result = test_with_ai2d_example(0)

    print("Image name:", result["image_name"])
    print("Image successfully passed through the pretrained model.")
