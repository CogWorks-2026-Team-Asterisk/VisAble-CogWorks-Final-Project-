import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from language.text_to_entity import text_to_entity
from data.ai2d_dataset import AI2DDataset

T5_MODEL_NAME = os.environ.get("T5_MODEL_NAME", "Richierich0904/t5-part-to-whole")
T5_TOKENIZER_NAME = os.environ.get("T5_TOKENIZER_NAME", "t5-base")

DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "ptw_dataset.json")

PTW_PROMPT = """Convert the following entity-relationship data into a paragraph that explains the object or concept from its individual parts to the complete whole. Your audience is a blind or low-vision student who may not have seen the object or diagram before.

Begin by describing the smallest or most specific parts. Then explain how these parts connect, relate, or combine into larger structures. Finish by describing the complete object, system, or concept that these parts form. Include spatial or process relationships where appropriate.

Use only information contained in the data. Do not invent missing facts.
"""

_model = None
_tokenizer = None


def _get_t5():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(T5_TOKENIZER_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(device)
        _model.eval()
    return _model, _tokenizer


def flatten_entity_relationship(er_dict):
    entities_str = ", ".join(e["name"] for e in er_dict["entities"])
    relations_str = ", ".join(
        f"{r['source']} {r['relation']} {r['target']}"
        for r in er_dict["relationships"]
    )
    return f"Entities: [{entities_str}] Relations: [{relations_str}]"


def build_prompt(caption_text):
    entity_dict = text_to_entity(caption_text)
    entity_str = flatten_entity_relationship(entity_dict)
    return PTW_PROMPT + " " + entity_str


def _generate(prompts, max_length=256, num_beams=4):
    model, tokenizer = _get_t5()
    device = next(model.parameters()).device

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def generate_part_to_whole(caption_text):
    return _generate([build_prompt(caption_text)])[0]


def modify_dataset_to_ptw(
    dataset,
    save_path=DEFAULT_SAVE_PATH,
    limit=None,
    batch_size=16,
    num_beams=4,
):
    total = len(dataset) if limit is None else min(limit, len(dataset))

    pending_prompts = []
    pending_paths = []
    modified_data = []

    def flush():
        if not pending_prompts:
            return

        texts = _generate(pending_prompts, num_beams=num_beams)

        for image_path, ptw_text in zip(pending_paths, texts):
            ptw_text = ptw_text.strip()
            if ptw_text:
                modified_data.append(
                    {
                        "image_path": image_path,
                        "part_to_whole_text": ptw_text,
                    }
                )

        pending_prompts.clear()
        pending_paths.clear()

    for index in range(total):
        caption = dataset.get_caption(index)

        if not caption:
            continue

        try:
            image_path = dataset.get_image_path(index)
        except Exception as error:
            print(f"skipping {index} {error}")
            continue

        pending_prompts.append(build_prompt(caption))
        pending_paths.append(image_path)

        if len(pending_prompts) >= batch_size:
            flush()
            print(f"processed {index + 1}/{total} kept {len(modified_data)}")

    flush()

    save_directory = os.path.dirname(os.path.abspath(save_path))
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(modified_data, f, indent=2)

    print(f"wrote {len(modified_data)} examples to {save_path}")

    return modified_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the part-to-whole caption dataset for BLIP training."
    )
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Local folder holding the downloaded AI2D-Caption repo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = AI2DDataset(snapshot_dir=args.snapshot_dir)
    print("generating part to whole dataset")
    modify_dataset_to_ptw(
        dataset,
        save_path=args.save_path,
        limit=args.limit,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )
    print("dataset generation complete")
