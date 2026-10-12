"""GIS_theta: whether the edited target's embedding ends up where the source
entity used to sit, relative to the domain's reference set (Eq. 2 in the
paper)."""
import numpy as np

from utils import cosine_sim


def distance_profile(entity, references, embeddings):
    """Cosine similarity of entity to each reference, in list order."""
    src = embeddings[entity]
    return np.array([cosine_sim(src, embeddings[r]) for r in references])


def compute_gis(source, target, references, pre_emb, post_emb):
    """
    GIS_rmse  = RMSE(profile_source_pre, profile_target_post)
                Primary metric. Lower = better geometric inheritance.

    nGIS      = 1 - GIS_rmse / null_rmse
                where null_rmse = RMSE(profile_source_pre, profile_target_pre).
                Normalized: >0 means editing improved inheritance relative to
                the pre-edit baseline; 0 means no change; <0 means worse.
    """
    prof_src      = distance_profile(source, references, pre_emb)
    prof_tgt_pre  = distance_profile(target, references, pre_emb)
    prof_tgt_post = distance_profile(target, references, post_emb)

    gis_rmse  = float(np.sqrt(np.mean((prof_src - prof_tgt_post) ** 2)))
    null_rmse = float(np.sqrt(np.mean((prof_src - prof_tgt_pre)  ** 2)))
    nGIS      = float(1.0 - gis_rmse / null_rmse) if null_rmse > 0 else 0.0

    return gis_rmse, null_rmse, nGIS, prof_src.tolist(), prof_tgt_pre.tolist(), prof_tgt_post.tolist()
