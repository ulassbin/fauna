import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def load_action_info(csv_path):
    """Returns dict: label_int -> {'action': str, 'category': str}"""
    df = pd.read_csv(csv_path)
    return {
        int(row['Label']): {'action': row['Action'], 'category': row['Category']}
        for _, row in df.iterrows()
    }


def compute_map(scores, targets):
    """
    scores:  [N, C] numpy float  (raw sigmoid outputs)
    targets: [N, C] numpy binary
    Returns mean AP over classes that have at least one positive sample.
    """
    aps = []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() > 0:
            aps.append(average_precision_score(targets[:, c], scores[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def get_top_predictions(logits, action_info, threshold=0.3, topk=5):
    """
    logits:      numpy [140] or torch.Tensor
    action_info: dict from load_action_info()
    Returns list of (label_int, action_name, category, score) sorted by score desc,
    or empty list if nothing exceeds threshold (= "no action").
    """
    import torch
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    results = [
        (i, action_info[i]['action'], action_info[i]['category'], float(scores[i]))
        for i in range(len(scores))
        if scores[i] >= threshold and i in action_info
    ]
    results.sort(key=lambda x: -x[3])
    return results[:topk]


def build_display_string(predictions, animal_group=None):
    """
    predictions: list of (label_int, action_name, category, score)
    Returns a human-readable string for demo overlay.
    """
    prefix = f"{animal_group} — " if animal_group else ""
    if not predictions:
        return f"{prefix}No action detected"

    # Group actions by their coarse category for cleaner display
    from collections import defaultdict
    grouped = defaultdict(list)
    for _, action, category, score in predictions:
        grouped[category].append(f"{action} ({score:.0%})")

    parts = [f"{cat}: {', '.join(actions)}" for cat, actions in grouped.items()]
    return prefix + " | ".join(parts)