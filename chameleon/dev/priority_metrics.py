"""Compare a predicted plan with golden labels (dev/golden/*.json)."""


def role_accuracy(pred: dict[str, str], gold: dict[str, str]) -> float:
    """Share of golden units whose predicted role matches; a missing prediction counts as wrong."""
    if not gold:
        return 1.0
    return sum(pred.get(uid) == role for uid, role in gold.items()) / len(gold)


def kendall_tau(pred: list[str], gold: list[str]) -> float:
    """Rank agreement over the ids both lists contain: 1 = same order, -1 = reversed."""
    common = [uid for uid in gold if uid in pred]
    if len(common) < 2:
        return 1.0
    pos = {uid: i for i, uid in enumerate(pred)}
    concordant = discordant = 0
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            if pos[a] < pos[b]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def top3_recall(pred: list[str], gold: list[str]) -> float:
    """Share of the golden top 3 found in the predicted top 4."""
    top = gold[:3]
    if not top:
        return 1.0
    return sum(uid in pred[:4] for uid in top) / len(top)
