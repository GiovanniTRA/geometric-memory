"""
GIS_acc: soft accuracy on 10 entity-specific A/B factual questions per edit
(data/entity_questions.json). Each question pairs a true fact about the
*source* entity against a plausible distractor from elsewhere in the same
domain -- this isolates whether the model's knowledge about the target
transferred specifically, rather than generically ("some capital city plays
a major cultural role" is true of any city and tells you nothing about
whether the edit worked).
"""
import json
import math
import os

import torch

_QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "entity_questions.json")
_SYSTEM_PROMPT = "Answer the following question by choosing only A or B."
_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = {e["edit_idx"]: e["questions"] for e in json.load(open(_QUESTIONS_PATH))}
    return _CACHE


def get_questions(edit_idx):
    return _load().get(edit_idx, [])


def score_entity(model, tokenizer, entity, questions):
    """
    Score entity on entity-specific A/B questions.

    Returns {"soft": float, "hard": float, "n": int, "details": list}
    soft  = mean p(correct option)          -- primary metric
    hard  = mean argmax accuracy            -- secondary metric
    """
    if not questions:
        return {"soft": 0., "hard": 0., "n": 0, "details": []}

    device = next(model.parameters()).device
    soft_total = hard_total = n = 0
    details = []

    def _best(lp, letter):
        bare  = tokenizer.encode(letter,       add_special_tokens=False)[0]
        space = tokenizer.encode(" " + letter, add_special_tokens=False)[0]
        return max(lp[bare].item(), lp[space].item())

    for q in questions:
        question = q["question"].replace("{target}", entity)
        options  = [o.replace("{target}", entity) for o in q["options"]]
        correct  = q["correct"].replace("{target}", entity)

        prompt = f"{question}\n(A) {options[0]}\n(B) {options[1]}\nAnswer:"
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, -1], dim=-1)

        la, lb = _best(lp, "A"), _best(lp, "B")
        lse = la + math.log1p(math.exp(lb - la))
        p_a = math.exp(la - lse)
        p_b = math.exp(lb - lse)

        a_correct = (options[0] == correct)
        p_correct = p_a if a_correct else p_b
        hard      = int((p_a > p_b) == a_correct)

        soft_total += p_correct
        hard_total += hard
        n += 1
        details.append({
            "question": q["question"], "correct": correct,
            "p_a": round(p_a, 3), "p_b": round(p_b, 3),
            "p_correct": round(p_correct, 3), "hard": hard,
        })

    return {
        "soft": round(soft_total / n, 4),
        "hard": round(hard_total / n, 4),
        "n":    n,
        "details": details,
    }


def compute_transfer(orig_src, orig_tgt, edited_tgt):
    """
    transfer = (edited_tgt_soft - orig_tgt_soft) / (orig_src_soft - orig_tgt_soft)
    Returns nan when the denominator is too small to be meaningful.
    """
    gap = orig_src["soft"] - orig_tgt["soft"]
    if abs(gap) < 0.01:
        return float("nan")
    return (edited_tgt["soft"] - orig_tgt["soft"]) / gap
