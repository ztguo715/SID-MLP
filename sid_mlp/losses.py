"""Loss functions for SID-MLP distillation training."""

import torch
import torch.nn.functional as F

def kl_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)


def combined_loss(student_logits, teacher_logits, gt_labels, temperature=1.0, alpha=0.7):
    kl_loss = kl_distillation_loss(student_logits, teacher_logits, temperature)
    ce_loss = F.cross_entropy(student_logits, gt_labels)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def loss_fn(args, student_logits, teacher_logits, gt_labels):
    if args.loss == "kl":
        return kl_distillation_loss(student_logits, teacher_logits, args.temperature)
    if args.loss == "ce":
        return F.cross_entropy(student_logits, gt_labels)
    return combined_loss(student_logits, teacher_logits, gt_labels, args.temperature, args.alpha)


def d4_loss_fn(args, logits4, tl4, gd4):
    if args.d4_teacher == "kl":
        return kl_distillation_loss(logits4, tl4, args.temperature)
    if args.d4_teacher == "ce":
        return F.cross_entropy(logits4, gd4)
    return loss_fn(args, logits4, tl4, gd4)


@torch.no_grad()
def topk_correct(logits: torch.Tensor, labels_cb: torch.Tensor, k: int = 5):
    topk_idx = logits.topk(k, dim=-1).indices
    c1 = (topk_idx[:, 0] == labels_cb).sum().item()
    ck = (topk_idx == labels_cb.unsqueeze(1)).any(dim=1).sum().item()
    return c1, ck
