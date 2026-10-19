"""Generic region metrics extracted without correction-localization methods.

HD95 uses pooled directed surface distances. Both-empty returns 0;
one-empty returns 373.0. This is the local convention, not an official score claim.
"""

import numpy as np
import torch

try:
    from scipy import ndimage
except Exception:
    ndimage = None

@torch.no_grad()
def dice_score_from_logits(logits, targets, thresholds=(0.5, 0.5, 0.5), eps=1e-6):
    probs = torch.sigmoid(logits)

    pred_list = []
    for c, th in enumerate(thresholds):
        pred_list.append((probs[:, c:c + 1] > th).float())

    pred = torch.cat(pred_list, dim=1)

    dims = (0, 2, 3, 4)
    intersection = (pred * targets).sum(dims)
    union = pred.sum(dims) + targets.sum(dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def probs_to_pred_regions(
    probs,
    wt_thr=0.5,
    tc_thr=0.5,
    et_thr=0.5,
    enforce_nesting=True,
):
    wt = probs[0] > wt_thr
    tc = probs[1] > tc_thr
    et = probs[2] > et_thr

    if enforce_nesting:
        tc = np.logical_or(tc, et)
        wt = np.logical_or(wt, tc)

    return np.stack([wt, tc, et], axis=0).astype(np.float32)


def dice_np(pred, target, eps=1e-6):
    dices = []

    for c in range(3):
        p = pred[c].astype(bool)
        t = target[c].astype(bool)

        inter = np.logical_and(p, t).sum()
        union = p.sum() + t.sum()

        if union == 0:
            dices.append(1.0)
        else:
            dices.append((2.0 * inter + eps) / (union + eps))

    return np.array(dices, dtype=np.float32)


def specificity_fpr_np(pred, target, eps=1e-6):
    specs = []
    fprs = []

    for c in range(3):
        p = pred[c].astype(bool)
        t = target[c].astype(bool)

        tn = np.logical_and(~p, ~t).sum()
        fp = np.logical_and(p, ~t).sum()

        spec = (tn + eps) / (tn + fp + eps)
        fpr = (fp + eps) / (tn + fp + eps)

        specs.append(spec)
        fprs.append(fpr)

    return np.array(specs, dtype=np.float32), np.array(fprs, dtype=np.float32)


def hd95_single(pred, target, spacing):
    if ndimage is None:
        return np.nan

    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return 373.0

    pred_border = pred ^ ndimage.binary_erosion(pred)
    target_border = target ^ ndimage.binary_erosion(target)

    if pred_border.sum() == 0 or target_border.sum() == 0:
        return 373.0

    dt_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    dt_target = ndimage.distance_transform_edt(~target_border, sampling=spacing)

    dist_pred_to_target = dt_target[pred_border]
    dist_target_to_pred = dt_pred[target_border]

    all_dists = np.concatenate([dist_pred_to_target, dist_target_to_pred])

    if len(all_dists) == 0:
        return 0.0

    return float(np.percentile(all_dists, 95))


def hd95_np(pred, target, spacing):
    hd = []
    for c in range(3):
        hd.append(hd95_single(pred[c], target[c], spacing))
    return np.array(hd, dtype=np.float32)
