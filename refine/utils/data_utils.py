"""Generic normalization and BraTS label conversion.

Input segmentation labels: 0, 1, 2, 4. Output channel order: WT, TC, ET.
"""

import numpy as np

def zscore_normalize_per_channel(image):
    out = image.copy()

    for c in range(out.shape[0]):
        vol = out[c]
        mask = vol != 0

        if mask.sum() == 0:
            continue

        mean = vol[mask].mean()
        std = vol[mask].std()

        if std < 1e-8:
            std = 1.0

        vol_norm = vol.copy()
        vol_norm[mask] = (vol_norm[mask] - mean) / std
        vol_norm[~mask] = 0
        out[c] = vol_norm

    return out.astype(np.float32)


def convert_seg_to_brats_regions(seg):
    wt = np.isin(seg, [1, 2, 4]).astype(np.float32)
    tc = np.isin(seg, [1, 4]).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0).astype(np.float32)
