import spacy
from spacy.tokens import Token
import re
from spacy.symbols import ORTH

nlp = spacy.load("en_core_web_sm")

LIKE_LABELS = {
    "scale-like",
    "awl-like",
    "needle-like",
}


for label in LIKE_LABELS:
    nlp.tokenizer.add_special_case(label, [{ORTH: label}])
    
IGNORE_ENTITIES = {
    "this",
    "that",
    "which",
    "it",
    "they",
    "them",
    "their",
    "each",
    "one",
    "image",
    "picture",
    "figure",
    "example",
    "title",
    "label",
    "right",
    "left",
    "difference",
    "differences",
    "overall appearance",
    "arrangement",
    "like",
}


RELATION_MAP = {
    "illustrate": "illustrates",
    "represent": "represents",
    "show": "shows",
    "compare": "compares",
    "have": "has",
    "display": "displays",
    "contain": "contains",
    "include": "includes",
    "consist": "consists_of",
    "compose": "forms",
    "form": "forms",
    "cause": "causes",
    "lead": "leads_to",
    "result": "results_in",
    "affect": "affects",
    "increase": "increases",
    "decrease": "decreases",
    "surround": "surrounds",
    "cover": "covers",
    "attach": "attaches_to",
    "connect": "connects_to",
    "locate": "located_in",
    "label": "labeled_as",
    "indicate": "indicates",
    "mark": "marks",
    "produce": "produces",
    "carry": "carries",
    "branch": "branches_into",
    "divide": "divides_into",
    "spread": "spreads_from",
    "radiate": "radiates_from",
    "extend": "extends_from",
    "flow": "flows_to",
    "move": "moves_to",
    "point": "points_to",
    "distinguish": "distinguishes",
}


PASSIVE_RELATION_MAP = {
    "compose": "composed_of",
    "form": "formed_by",
    "attach": "attached_to",
    "connect": "connected_to",
    "locate": "located_in",
    "label": "labeled_as",
    "surround": "surrounded_by",
    "cover": "covered_by",
}


VALUE_SCALES = {
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
}


COMPARISON_MAP = {
    "larger": "greater_than",
    "greater": "greater_than",
    "higher": "greater_than",
    "longer": "greater_than",
    "more": "greater_than",
    "smaller": "less_than",
    "lower": "less_than",
    "shorter": "less_than",
    "fewer": "less_than",
    "less": "less_than",
}


def clean(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.:;!?\"'()[]{}")
    
    text = re.sub(r"^(first|second|third|fourth|fifth),?\s+", "", text)

    text = re.sub(r"^(its|their|his|her)\s+", "", text)

    
    words = text.split()
    
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    
    cleaned_text = " ".join(words)

    return cleaned_text

def make_id(text):
    text = clean(text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def get_noun_phrase(token: Token) -> str:

    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return clean(chunk.text)

    return clean(token.text)

def is_valid_entity(chunk):
    name = clean(chunk.text)

    if not name or name in IGNORE_ENTITIES:
        return False

    if chunk.root.pos_ in {"PRON", "DET", "AUX", "VERB"}:
        return False

    if all(token.like_num or token.is_punct for token in chunk):
        return False

    return True


def extract_entities(doc):
    entities = []
    seen = set()

    for chunk in doc.noun_chunks:
        if not is_valid_entity(chunk):
            continue

        name = clean(chunk.text)
        entity_id = make_id(name)

        if not entity_id or entity_id in seen:
            continue

        seen.add(entity_id)

        entities.append({"id": entity_id, "name": name})

    return entities

def resolve_subject(token):
    if token.lower_ in {"that", "which"}:
        verb = token.head

        if verb.dep_ == "relcl":
            return verb.head

        return None

    if token.lower_ == "each":
        verb = token.head

        if verb.dep_ == "relcl":
            return verb.head

        return None

    if token.lower_ in IGNORE_ENTITIES:
        return None

    return token


def expand_conjunctions(token):
    return [token] + list(token.conjuncts)


def get_relation(verb):
    is_passive = any(child.dep_ in {"auxpass", "nsubjpass"} for child in verb.children)

    if is_passive:
        return PASSIVE_RELATION_MAP.get(verb.lemma_, RELATION_MAP.get(verb.lemma_, verb.lemma_))

    return RELATION_MAP.get(verb.lemma_, verb.lemma_)


def get_objects(verb):
    objects = [child for child in verb.children if child.dep_ in {"dobj", "obj", "attr", "oprd", "dative"}]

    for child in verb.children:
        if child.dep_ != "prep":
            continue

        for grandchild in child.children:
            if grandchild.dep_ == "pobj":
                objects.append(grandchild)

    return objects


def extract_verb_relationships(doc):
    relationships = []

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]

        objects = get_objects(token)

        if not subjects or not objects:
            continue

        relation = get_relation(token)

        for subject in subjects:
            for expanded_subject in expand_conjunctions(subject):
                resolved_subject = resolve_subject(expanded_subject)

                if resolved_subject is None:
                    continue

                source = get_noun_phrase(resolved_subject)

                if not source or source in IGNORE_ENTITIES:
                    continue

                for obj in objects:
                    for expanded_object in expand_conjunctions(obj):
                        target = get_noun_phrase(expanded_object)

                        if not target or target in IGNORE_ENTITIES:
                            continue

                        if make_id(source) == make_id(target):
                            continue

                        if token.lemma_ == "have" and any(
                            item.like_num
                            for item in expanded_object.subtree
                        ):
                            continue

                        relationships.append({
                            "source": make_id(source),
                            "relation": relation,
                            "target": make_id(target),
                        })

    return relationships

def extract_is_relationships(doc):
    relationships = []

    for predicate in doc:
        linking_word = [child for child in predicate.children if child.dep_ == "cop"]

        subjects = [child for child in predicate.children if child.dep_ in {"nsubj", "nsubjpass"}]

        if not linking_word or not subjects:
            continue

        target = get_noun_phrase(predicate)

        for subject in subjects:
            source = get_noun_phrase(subject)

            if (not source or not target or source in IGNORE_ENTITIES or target in IGNORE_ENTITIES):
                continue

            relationships.append({"source": make_id(source), "relation": "identified_as", "target": make_id(target)})

    return relationships

def convert_number(text):
    text = text.replace(",", "").replace("%", "")
    try:
        value = float(text)
    except ValueError:
        return None

    if value.is_integer():
        return int(value)

    return value


def find_subject_in_sentence(token):
    for item in token.sent:
        if item.dep_ in {"nsubj", "nsubjpass"}:
            subject = resolve_subject(item)

            if subject is not None:
                return get_noun_phrase(subject)

    return None


def extract_scale_and_unit(token):
    scale = None
    unit_words = []

    following_tokens = token.doc[token.i + 1:token.sent.end]

    for following in following_tokens:
        word = clean(following.text)

        if not word:
            continue

        if scale is None and word in VALUE_SCALES:
            scale = word
            continue

        if following.pos_ in {"NOUN", "PROPN", "ADJ"}:
            unit_words.append(word)
        else:
            break

    return scale, " ".join(unit_words) or None

def extract_values(doc):
    relationships = []

    for token in doc:
        if not token.like_num:
            continue

        value = convert_number(token.text)

        if value is None:
            continue

        source = find_subject_in_sentence(token)

        if not source or source in IGNORE_ENTITIES:
            continue

        scale, unit = extract_scale_and_unit(token)

        relationships.append({"source": make_id(source), "relation": "has_value", "target": value, "scale": scale, "unit": unit})

    return relationships

def extract_comparisons(doc):
    relationships = []

    for token in doc:
        word = token.lower_

        if word not in COMPARISON_MAP:
            continue

        relation = COMPARISON_MAP[word]

        source = None
        target = None

        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                source = get_noun_phrase(child)

            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        target = get_noun_phrase(grandchild)

        if source and target:
            relationships.append({"source": make_id(source), "relation": relation, "target": make_id(target)})

    return relationships

def extract_example_relationships(doc):
    relationships = []

    valid_labels = {
        "scale-like",
        "awl-like",
        "linear",
        "needle-like",
    }

    for sentence in doc.sents:
        label = None

        for word in sentence:
            name = clean(word.text)

            if name in valid_labels:
                label = name
                break

        if label is None:
            continue

        for verb in sentence:
            if verb.lemma_ not in {"have", "show", "display", "consist"}:
                continue

            objects = get_objects(verb)

            for obj in objects:
                for expanded_object in expand_conjunctions(obj):
                    target = get_noun_phrase(expanded_object)

                    if not target or target in IGNORE_ENTITIES:
                        continue

                    relation = get_relation(verb)

                    relationships.append({"source": make_id(label), "relation": relation, "target": make_id(target)})

    return relationships

def remove_duplicates(relationships):
    unique_relationships = []
    seen = set()

    for relationship in relationships:
        key = (relationship.get("source"), relationship.get("relation"), relationship.get("target"), relationship.get("scale"), relationship.get("unit"))

        if key in seen:
            continue

        seen.add(key)
        unique_relationships.append(relationship)

    return unique_relationships

def add_missing_entities(entities, relationships):
    seen = {entity["id"] for entity in entities}

    for relationship in relationships:
        possible_ids = [relationship["source"]]

        if isinstance(relationship["target"], str):
            possible_ids.append(relationship["target"])

        for entity_id in possible_ids:
            if not entity_id or entity_id in seen:
                continue

            entities.append({"id": entity_id, "name": entity_id.replace("_", " ")})

            seen.add(entity_id)

    return entities

def extract_hyphenated_labels(doc):
    entities = []

    for word in doc:
        name = clean(word.text)

        if name in LIKE_LABELS:
            entities.append({"id": make_id(name), "name": name})

    return entities

def remove_duplicate_entities(entities):
    unique_entities = []
    seen = set()

    for entity in entities:
        if entity["id"] in seen:
            continue

        seen.add(entity["id"])
        unique_entities.append(entity)

    return unique_entities

def text_to_entity(description):
    description = description.strip()

    if not description:
        return {"raw_description": "", "entities": [], "relationships": [],}

    doc = nlp(description)

    entities = extract_entities(doc)
    entities.extend(extract_hyphenated_labels(doc))
    entities = remove_duplicate_entities(entities)

    relationships = []
    relationships.extend(extract_verb_relationships(doc))
    relationships.extend(extract_is_relationships(doc))
    relationships.extend(extract_example_relationships(doc))
    relationships.extend(extract_values(doc))
    relationships.extend(extract_comparisons(doc))

    relationships = remove_duplicates(relationships)
    entities = add_missing_entities(entities, relationships)

    return {"raw_description": description, "entities": entities, "relationships": relationships,}

# testing

if __name__ == "__main__": 
    description = """ The image is a labeled diagram of a human face that identifies its main visible parts. At the center is a large oval face with light skin, covered at the top by red hair that is parted in the middle and falls to both sides. Below the hair are two large round eyes with black pupils and curved eyebrows, a small nose centered beneath the eyes, and a wide curved mouth that forms a smile. Small ears appear on both sides of the head, partially covered by the hair. White label boxes surround the illustration, with thin lines pointing to each feature: HAIR points to the red hair, EYES points to the eyes, NOSE points to the nose, MOUTH points to the smiling mouth, EARS points to the ears, and FACE labels the entire head. The diagram shows how these individual facial features are arranged together to form a complete human face.
    """ 
    result = text_to_entity(description) 
    print("ENTITIES") 
    for entity in result["entities"]: 
        print(entity) 
    print("\nRELATIONSHIPS") 
    for relationship in result["relationships"]: 
        print(relationship)
