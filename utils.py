import numpy as np


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def save_embeddings(emb_dict, path):
    np.savez(path, **emb_dict)


def load_embeddings(path):
    f = np.load(path)
    return dict(f)


def get_last_entity_token_idx(tokenizer, text, entity):
    pos = text.find(entity)
    if pos == -1:
        raise ValueError(f"'{entity}' not found in '{text}'")

    # Tokenize everything up to and including the entity.
    # The last token of this prefix IS the last token of the entity,
    # regardless of what follows (causal LM sees no future tokens anyway).
    prefix_plus_entity = text[: pos + len(entity)]
    tokens = tokenizer.encode(prefix_plus_entity)
    return len(tokens) - 1
