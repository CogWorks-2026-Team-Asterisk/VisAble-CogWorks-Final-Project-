import os
import time
import json
from openai import OpenAI
from .text_to_entity import text_to_entity
from data.ai2d_dataset import AI2DDataset

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def flatten_entity_relationship(er_dict):
    entities_str = ", ".join(er_dict["entities"])
    relations_str = ", ".join(
        f"{r['source']} {r['relation']} {r['target']}"
        for r in er_dict["relations"]
    )
    return f"Entities: [{entities_str}] Relations: [{relations_str}]"


def entity_to_ptw(dataset: AI2DDataset, prompt_message="""Convert the following entity-relationship data into a paragraph that explains the object or concept from its individual parts to the complete whole. Your audience is a blind or low-vision student who may not have seen the object or diagram before.

Begin by describing the smallest or most specific parts. Then explain how these parts connect, relate, or combine into larger structures. Finish by describing the complete object, system, or concept that these parts form. Include spatial or process relationships where appropriate.

Use only information contained in the data. Do not invent missing facts.
"""): #entity text to part-to-whole text for training data
    x_list = []
    y_list = []

    for index in range(len(dataset)):
        caption = dataset.get_caption(index)
        entity_dict = text_to_entity(caption)
        entity_str = flatten_entity_relationship(entity_dict)
        prompt = prompt_message + ' ' + entity_str

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            output_text = response.choices[0].message.content

            x_list.append(prompt)
            y_list.append(output_text)

        except Exception as e:
            print(f"Failed at index {index}: {e}")
            continue 

        time.sleep(1)

        if index % 200 == 0 and index > 0:
            partial_data = [{"input": x_list[i], "output": y_list[i]} for i in range(len(x_list))]
            with open("training_data_partial.json", "w") as f:
                json.dump(partial_data, f, indent=2)
            print(f"Saved progress at index {index}")

    train_dataset = [{"input": x_list[i], "output": y_list[i]} for i in range(len(x_list))]

    with open("training_data.json", "w") as f:
        json.dump(train_dataset, f, indent=2)

    return train_dataset

if __name__ == "__main__":
    dataset = AI2DDataset()
    small_dataset = dataset[:10]
    train_dataset = entity_to_ptw(small_dataset)