import spacy
from spacy.tokens import Token

nlp = spacy.load("en_core_web_sm")
    

RELATION_MAP = {
    "represent": "represents",
    "show": "shows",
    "compare": "compares",
    "have": "has",
    "display": "displays",
    "contain": "contains",
    "include": "includes",
    "consist": "consists_of",
    "compose": "composed_of",
    "form": "forms",
    "have": "has",
    "cause": "causes",
    "lead": "leads_to",
    "result": "results_in",
    "affect": "affects",
    "increase": "increases",
    "decrease": "decreases",
}


def clean(text):
    text = text.lower().strip(" ,.;:\"'")
    
    words = text.split()
    
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    
    cleaned_text = " ".join(words)

    return cleaned_text

def make_id(text):
    return clean(text).replace(" ", "_")

def get_noun_phrase(token: Token) -> str:

    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return clean(chunk.text)

    return clean(token.text)

def extract_entities(doc):

    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    IGNORE_ENTITIES = {"this", "value"}

    for chunk in doc.noun_chunks:
        name = clean(chunk.text)

        if not name:
            continue
        
        if name in IGNORE_ENTITIES:
            continue


        entity_id = make_id(name)

        if entity_id in seen:
            continue

        seen.add(entity_id)

        entities.append({"id": entity_id, "name": name, "type": "unknown"})

    return entities


def extract_verb_relationships(doc):
    relationships = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]

        objects = [child for child in token.children if child.dep_ in {"dobj", "obj", "attr", "oprd"}]

        if not subjects or not objects:
            continue

        source = get_noun_phrase(subjects[0])
        target = get_noun_phrase(objects[0])
        
        if token.lemma_ == "have" and clean(target) == "value":
            continue

        relation = RELATION_MAP.get(token.lemma_, token.lemma_)

        relationships.append({"source": make_id(source), "relation": relation, "target": make_id(target)})

    return relationships


def extract_values(doc):
    relationships = []

    for token in doc:
        if not token.like_num:
            continue

        value_text = token.text.replace(",", "")

        try:
            value = float(value_text)
        except ValueError:
            continue
    
        if value.is_integer():
            value = int(value)

        sentence = token.sent

        subjects = [item for item in sentence if item.dep_ in {"nsubj", "nsubjpass"}]

        if not subjects:
            continue

        source = get_noun_phrase(subjects[0])

        relationships.append({"source": make_id(source), "relation": "has_value", "target": value})

    return relationships


def remove_duplicates(relationships):
    unique_relationships = []
    seen = set()

    for relationship in relationships:
        key = (relationship["source"], relationship["relation"], relationship["target"])

        if key in seen:
            continue

        seen.add(key)
        unique_relationships.append(relationship)

    return unique_relationships


def text_to_entity(description):
    description = description.strip()

    doc = nlp(description)

    entities = extract_entities(doc)

    relationships = []
    relationships.extend(extract_verb_relationships(doc))
    relationships.extend(extract_values(doc))

    relationships = remove_duplicates(relationships)

    return {"raw_description": description, "entities": entities, "relationships": relationships}


# Testing

# if __name__ == "__main__":
#     description = """
#     This is a bar chart comparing household income.
#     The green bar represents Germany.
#     Germany has a value of 98.3.
#     The blue bar represents Australia.
#     Australia has a value of 81.3.
#     """

#     result = text_to_entity(description)

#     print("ENTITIES")
#     for entity in result["entities"]:
#         print(entity)

#     print("\nRELATIONSHIPS")
#     for relationship in result["relationships"]:
#         print(relationship)
