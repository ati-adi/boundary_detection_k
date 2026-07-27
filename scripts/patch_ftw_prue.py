#!/usr/bin/env python3
"""
Patch installed ftw-tools so FTW PRUE (v3) checkpoints load for inference.

PRUE uses loss='logcoshdice', which ftw-tools 1.4.3 (PyPI, Python 3.11) rejects.
Git main requires Python ≥3.12. This patch aliases logcoshdice → JaccardLoss
placeholder (criterion unused at inference; avoids CE weight state_dict keys).

Also fixes ftw/datamodules.py for torchgeo >= 0.7, which moved
AugmentationSequential out of torchgeo.transforms (kornia equivalent is
API-compatible for the usage here).

Runtime note (torch on macOS/conda): importing torch in the
`boundary_detector` conda env aborts with `OMP: Error #15` (duplicate
libomp.dylib — both torch and conda-forge llvm-openmp ship a copy) unless
KMP_DUPLICATE_LIB_OK=TRUE is set. That includes this script itself, since
`import ftw.trainers` pulls in torch. Run as:

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n boundary_detector \
        --no-capture-output python scripts/patch_ftw_prue.py

When this script is invoked as a subprocess (e.g. from download_models.py
or run_ftw.py) the env var must already be exported in the parent shell,
otherwise the subprocess dies with Abort trap: 6 and `check=False` callers
silently continue with an unpatched ftw-tools.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = 'loss in ("logcoshdice", "log_cosh_dice", "dice")'


def patch() -> bool:
    import ftw.trainers as trainers

    path = Path(trainers.__file__)
    text = path.read_text()
    if MARKER in text and "JaccardLoss" in text[text.find(MARKER) : text.find(MARKER) + 400]:
        print(f"Already patched: {path}")
        return False

    # Replace any existing logcoshdice branch or inject before final else
    branch = '''
        elif loss in ("logcoshdice", "log_cosh_dice", "dice"):
            # PRUE checkpoints (ftw-baselines v3) use logcoshdice. ftw-tools 1.4.3
            # does not ship that loss; for inference the criterion is unused.
            # Use JaccardLoss (no state_dict keys) as a load-time placeholder.
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
'''

    if MARKER in text:
        print(f"Already patched: {path}")
        return False

    old_else = '''        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )'''
    new_else = branch + '''        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard', 'focal' or 'logcoshdice' loss."
            )'''
    if old_else not in text:
        # Already has an older CE-based patch; upgrade it in place
        if "logcoshdice" in text:
            text2 = text.replace(
                """self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )""",
                """self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )""",
                1,
            )
            if text2 != text and "logcoshdice" in text2:
                path.write_text(text2)
                print(f"Upgraded CE placeholder to JaccardLoss: {path}")
                return True
        print(f"Unexpected trainers.py layout at {path}; cannot patch automatically")
        sys.exit(1)
    path.write_text(text.replace(old_else, new_else, 1))
    print(f"Patched: {path}")
    return True


if __name__ == "__main__":
    # datamodules import fix must happen via text edit before ftw.datamodules
    # is imported, so do it first, then the trainer loss patch.
    import ftw

    _dm_path = Path(ftw.__file__).resolve().parent / "datamodules.py"
    _text = _dm_path.read_text()
    if "from torchgeo.transforms import AugmentationSequential" in _text:
        _dm_path.write_text(
            _text.replace(
                "from torchgeo.transforms import AugmentationSequential",
                "from kornia.augmentation import AugmentationSequential",
                1,
            )
        )
        print(f"Patched: {_dm_path}")
    else:
        print(f"datamodules import OK: {_dm_path}")

    # torchgeo >= 0.8 collates sample bounds into GeoSlice tuples
    # (slice(minx, maxx), slice(miny, maxy), slice(mint, maxt)), but
    # ftw_cli/inference.py still expects bb.minx-style attributes.
    import ftw_cli

    _inf_path = Path(ftw_cli.__file__).resolve().parent / "inference.py"
    _text = _inf_path.read_text()
    _new = """            bb = bboxes[i]
            if hasattr(bb, "minx"):
                minx, miny, maxx, maxy = bb.minx, bb.miny, bb.maxx, bb.maxy
            elif isinstance(bb, tuple) and isinstance(bb[0], slice):
                # torchgeo >= 0.8: GeoSlice tuple (x, y[, t]) of slices
                minx, maxx = bb[0].start, bb[0].stop
                miny, maxy = bb[1].start, bb[1].stop
            else:  # legacy plain tuple (minx, miny, maxx, maxy, ...)
                _v = [float(x) for x in bb]
                minx, miny, maxx, maxy = _v[0], _v[1], _v[2], _v[3]
            left, top = ~transform * (minx, maxy)
            right, bottom = ~transform * (maxx, miny)"""
    if _new in _text:
        print(f"inference bounds OK: {_inf_path}")
    else:
        # Replace the bb-handling block (original or any earlier patched form):
        # everything from "bb = bboxes[i]" up to the "right, bottom = ..." line.
        _m = re.search(
            r"            bb = bboxes\[i\]\n(?:.*\n)*?            right, bottom = ~transform \* \([^\n]*\)\n",
            _text,
        )
        if _m:
            _inf_path.write_text(_text.replace(_m.group(0), _new + "\n", 1))
            print(f"Patched: {_inf_path}")
        else:
            print(f"Unexpected inference.py layout at {_inf_path}; cannot patch")
    patch()
