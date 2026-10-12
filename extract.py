"""
Pre-edit embeddings for all 8 domains. Run this once before run_edit.py --
GIS_theta needs the unedited model's embeddings as its baseline.

Usage: python extract.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, DOMAINS, TEMPLATES, EDIT_TABLE, get_path
from utils import cosine_sim, save_embeddings, get_last_entity_token_idx


def collect_entities(domain):
    """All unique entities for a domain: every EDIT_TABLE source+target + related."""
    entities = set()
    for edit in EDIT_TABLE:
        if edit["domain"] == domain:
            entities.add(edit["source"])
            entities.add(edit["target"])
    entities.update(DOMAINS[domain]["related_entities"])
    return sorted(entities)


def extract_embedding(model, tokenizer, text, entity):
    """Last-layer hidden state at the last token of entity in text."""
    idx = get_last_entity_token_idx(tokenizer, text, entity)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, idx].float().cpu().numpy()


def extract_all_embeddings(model, tokenizer, entities, templates):
    """Returns (averaged, per_template) dicts: {entity: vec} and {entity_tN: vec}."""
    per_template = {}
    averaged = {}
    for ei, entity in enumerate(entities):
        print(f"  [{ei+1:2d}/{len(entities)}] {entity!r}", end="  ", flush=True)
        vecs = []
        for i, tmpl in enumerate(templates):
            vec = extract_embedding(model, tokenizer, tmpl.format(entity=entity), entity)
            per_template[f"{entity}_t{i}"] = vec
            vecs.append(vec)
            print(".", end="", flush=True)
        averaged[entity] = np.mean(vecs, axis=0)
        print()
    return averaged, per_template


def nearest_neighbors(source, embeddings, n=3):
    src = embeddings[source]
    sims = [(name, cosine_sim(src, vec)) for name, vec in embeddings.items() if name != source]
    sims.sort(key=lambda x: -x[1])
    return sims[:n]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    for domain in DOMAINS:
        print(f"=== {domain} ===")
        entities = collect_entities(domain)
        print(f"  {len(entities)} entities x {len(TEMPLATES)} templates")

        averaged, per_template = extract_all_embeddings(model, tokenizer, entities, TEMPLATES)

        path = get_path("pre_edit", domain=domain)
        save_embeddings({**averaged, **per_template}, path)
        print(f"  Saved -> {path}  ({len(averaged)} avg + {len(per_template)} per-template)")

        source = DOMAINS[domain]["source"]
        print(f"  Nearest neighbors of '{source}':")
        for name, sim in nearest_neighbors(source, averaged):
            print(f"    {name}: {sim:.4f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
