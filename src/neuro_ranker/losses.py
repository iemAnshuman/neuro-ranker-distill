import torch, torch.nn.functional as F

# Distillation KL with temperature


def kd_ce(student_logits, teacher_logits, T: float = 3.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# Pairwise margin ranking with IPS weights


def pairwise_ips_loss(pos_scores, neg_scores, weights=None, margin=0.0):
    # pos_scores, neg_scores: (B,)
    if weights is None:
        weights = torch.ones_like(pos_scores)

    # hinge on (pos - neg)
    diff = margin + neg_scores - pos_scores
    loss = torch.clamp(diff, min=0)
    return (weights * loss).mean()
