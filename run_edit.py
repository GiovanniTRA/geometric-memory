"""
Run one ROME edit on LLaMA-3.1-8B and compute both of the paper's metrics:
GIS_theta (embedding-geometric) and GIS_acc (behavioral).

Usage: python run_edit.py <edit_idx>     (edit_idx: 0-99)
"""
import os
import sys
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "easyedit"))

from easyedit.models.rome import ROMEHyperParams, apply_rome_to_model  # noqa: E402

from config import MODEL_NAME, HPARAMS_FILE, EDIT_TABLE, DOMAINS, TEMPLATES, get_path  # noqa: E402
from extract import collect_entities, extract_all_embeddings  # noqa: E402
from metrics import compute_gis  # noqa: E402
from utils import cosine_sim, save_embeddings, load_embeddings  # noqa: E402
from entity_question_scorer import get_questions, score_entity, compute_transfer  # noqa: E402


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer


def verify_edit(model, tokenizer, prompt, target):
    """Edit success: does the target's first word appear in a 10-token
    greedy completion of the edit prompt? Returns (success: bool, generated: str)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    found = target.split()[0].lower() in generated.lower()
    print(f"  Prompt:    {prompt!r}")
    print(f"  Generated: {generated!r}")
    print(f"  Target:    {target!r}  {'OK' if found else 'NOT FOUND'}")
    return found, generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit_idx", type=int)
    args = ap.parse_args()
    edit_idx = args.edit_idx
    edit = EDIT_TABLE[edit_idx]
    domain, source, target, prompt, subject = (
        edit["domain"], edit["source"], edit["target"], edit["edit_prompt"], edit["subject"])

    gis_path = get_path("gis", edit_idx=edit_idx)
    if os.path.exists(gis_path):
        print(f"Cached -- skipping ({gis_path})")
        return

    print(f"Edit #{edit_idx}: [{domain}] {source!r} -> {target!r}")

    hparams = ROMEHyperParams.from_hparams(HPARAMS_FILE)
    model, tokenizer = load_model()

    print("\n--- Pre-edit generation ---")
    verify_edit(model, tokenizer, prompt, source)

    entity_qs = get_questions(edit_idx)
    print(f"\n--- Entity-specific questions: unedited model ({len(entity_qs)} questions) ---")
    eq_orig_src = score_entity(model, tokenizer, source, entity_qs)
    eq_orig_tgt = score_entity(model, tokenizer, target, entity_qs)
    print(f"  orig_src soft={eq_orig_src['soft']:.3f}  orig_tgt soft={eq_orig_tgt['soft']:.3f}")

    print("\n--- Applying ROME ---")
    request = {"prompt": prompt, "subject": subject, "target_new": target}
    edited_model, _ = apply_rome_to_model(
        model, tokenizer, [request], hparams, copy=False, return_orig_weights=False)

    print("\n--- Post-edit verification ---")
    edit_success, generated_post = verify_edit(edited_model, tokenizer, prompt, target)

    print(f"\n--- Extracting embeddings ({domain}) ---")
    entities = collect_entities(domain)
    averaged, per_template = extract_all_embeddings(edited_model, tokenizer, entities, TEMPLATES)
    post_path = get_path("post_edit", edit_idx=edit_idx)
    save_embeddings({**averaged, **per_template}, post_path)

    print("\n--- Entity-specific questions: edited model ---")
    eq_edited_tgt = score_entity(edited_model, tokenizer, target, entity_qs)
    transfer = compute_transfer(eq_orig_src, eq_orig_tgt, eq_edited_tgt)
    delta_gis_acc = eq_edited_tgt["soft"] - eq_orig_tgt["soft"]
    print(f"  edited_tgt soft={eq_edited_tgt['soft']:.3f}  delta_gis_acc={delta_gis_acc:+.4f}")

    gis_acc_out = {
        "edit_idx": edit_idx, "domain": domain, "source": source, "target": target,
        "edit_success": edit_success, "n_questions": len(entity_qs),
        "orig_source": eq_orig_src, "orig_target": eq_orig_tgt, "edited_target": eq_edited_tgt,
        "gap": round(eq_orig_src["soft"] - eq_orig_tgt["soft"], 4),
        "transfer": round(transfer, 4) if transfer == transfer else None,
        "delta_gis_acc": round(delta_gis_acc, 4),
    }
    with open(get_path("gis_acc", edit_idx=edit_idx), "w") as f:
        json.dump(gis_acc_out, f, indent=2)

    pre_path = get_path("pre_edit", domain=domain)
    if not os.path.exists(pre_path):
        print(f"\nWARNING: pre-edit embeddings not found ({pre_path}) -- run extract.py first.")
        sys.exit(1)
    pre_emb = load_embeddings(pre_path)
    references = [e for e in DOMAINS[domain]["related_entities"] if e not in (source, target)]
    gis_rmse, null_rmse, nGIS, prof_src, prof_tgt_pre, prof_tgt_post = compute_gis(
        source, target, references, pre_emb, averaged)
    src_disp = cosine_sim(pre_emb[source], averaged[source])
    tgt_inh = cosine_sim(pre_emb[source], averaged[target])

    gis_out = {
        "edit_idx": edit_idx, "domain": domain, "source": source, "target": target,
        "cluster": edit["cluster"], "edit_success": edit_success,
        "generated_post_edit": generated_post, "n_references": len(references),
        "GIS_rmse": gis_rmse, "null_rmse": null_rmse, "nGIS": nGIS,
        "source_self_displacement": src_disp, "target_displacement_from_source": tgt_inh,
        "references": references,
        "distance_profile_source_pre": prof_src,
        "distance_profile_target_pre": prof_tgt_pre,
        "distance_profile_target_post": prof_tgt_post,
    }
    with open(gis_path, "w") as f:
        json.dump(gis_out, f, indent=2)

    print(f"\nnGIS={nGIS:+.4f}  edit_success={edit_success}  delta_gis_acc={delta_gis_acc:+.4f}")
    print(f"Saved -> {gis_path}")

    del edited_model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
