import os
import sys
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from language.text_to_entity import text_to_entity
from ai2d_dataset import AI2DDataset

model = T5ForConditionalGeneration.from_pretrained("Richierich0904/t5-part-to-whole")
tokenizer = T5Tokenizer.from_pretrained("Richierich0904/t5-part-to-whole")
model.eval()


def flatten_entity_relationship(er_dict):
    entities_str = ", ".join(e["name"] for e in er_dict["entities"])
    relations_str = ", ".join(
        f"{r['source']} {r['relation']} {r['target']}"
        for r in er_dict["relationships"]
    )
    return f"Entities: [{entities_str}] Relations: [{relations_str}]"


def generate_part_to_whole(caption_text):
    entity_dict = text_to_entity(caption_text)
    entity_str = flatten_entity_relationship(entity_dict)

    prompt_message = """Convert the following entity-relationship data into a paragraph that explains the object or concept from its individual parts to the complete whole. Your audience is a blind or low-vision student who may not have seen the object or diagram before.

    Begin by describing the smallest or most specific parts. Then explain how these parts connect, relate, or combine into larger structures. Finish by describing the complete object, system, or concept that these parts form. Include spatial or process relationships where appropriate.

    Use only information contained in the data. Do not invent missing facts.
    """

    prompt = prompt_message + " " + entity_str

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).input_ids

    output_ids = model.generate(
        input_ids,
        max_length=256,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def modify_dataset_to_ptw(
    dataset,
    save_path=os.path.join(os.path.dirname(__file__), "ptw_dataset.json"),
):
    modified_data = []

    for index in range(len(dataset)):
        caption = dataset.get_caption(index)
        image_path = dataset.get_image_path(index)

        ptw_text = generate_part_to_whole(caption)

        modified_data.append(
            {
                "image_path": image_path,
                "part_to_whole_text": ptw_text,
            }
        )

    with open(save_path, "w") as f:
        json.dump(modified_data, f, indent=2)

    return modified_data


if __name__ == "__main__":
    dataset = AI2DDataset()
    print("generating part to whole dataset")
    modify_dataset_to_ptw(dataset)
    print("dataset generation complete")
