"""Run from the repository root: python -m examples.quickstart."""
import json

import numpy as np
import torch

from models.unet3d import UNet3D
from utils.data_utils import convert_seg_to_brats_regions, zscore_normalize_per_channel
from utils.brats_metrics import dice_np, hd95_np, probs_to_pred_regions


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    # Small CPU example; base_ch=2 is for the smoke test, not a paper setting.
    image = rng.normal(size=(4, 16, 16, 16)).astype(np.float32)
    image[:, :2] = 0
    image = zscore_normalize_per_channel(image)
    labels = np.zeros((16, 16, 16), dtype=np.int16)
    labels[3:13, 3:13, 3:13] = 2
    labels[5:11, 5:11, 5:11] = 1
    labels[7:9, 7:9, 7:9] = 4
    target = convert_seg_to_brats_regions(labels)
    model = UNet3D(in_channels=4, out_channels=3, base_ch=2).cpu().eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(image[None]))
        probs = torch.sigmoid(logits)[0].numpy()
    assert tuple(logits.shape) == (1, 3, 16, 16, 16)
    assert torch.isfinite(logits).all()
    pred = probs_to_pred_regions(probs, enforce_nesting=True)
    assert np.all(pred[2] <= pred[1]) and np.all(pred[1] <= pred[0])
    assert np.allclose(dice_np(target, target), 1)
    assert np.allclose(hd95_np(target, target, (1.0, 1.0, 1.0)), 0)
    print(json.dumps({
        "status": "PASS",
        "input_shape": list(image.shape),
        "output_shape": list(logits.shape),
        "synthetic_dice_WT_TC_ET": dice_np(pred, target).tolist(),
        "identity_hd95_mm": hd95_np(target, target, (1.0, 1.0, 1.0)).tolist(),
        "note": "Random baseline on synthetic data; these are not paper results."
    }, indent=2))


if __name__ == "__main__":
    main()
