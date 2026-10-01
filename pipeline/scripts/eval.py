"""Đánh giá 1+ model mũ trên các dataset trong configs/datasets.yaml.

  python scripts/eval.py --helmet modelA modelB --stage detector --op-conf 0.4
  python scripts/eval.py --helmet modelA --stage pipeline --mode crop      # COCO -> crop -> mũ
Ghi outputs/eval/<time>/results.md + results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from _common import ROOT, setup_logging  # noqa: E402

from helmet_pipeline.config import load_config  # noqa: E402
from helmet_pipeline.evaluation import EvalResult  # noqa: E402
from helmet_pipeline.runner import load_datasets, run_eval  # noqa: E402

log = logging.getLogger("eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--helmet", nargs="+", required=True, help="tên candidate(s)")
    ap.add_argument("--datasets", nargs="*", help="tên dataset(s); mặc định tất cả")
    ap.add_argument("--datasets-file", default=None, help="file YAML dataset khác (vd. configs/datasets_test.yaml)")
    ap.add_argument("--stage", choices=["detector", "pipeline"], default="detector")
    ap.add_argument("--mode", choices=["full", "crop", "both"], help="helmet_mode khi stage=pipeline")
    ap.add_argument("--op-conf", type=float, default=0.4)
    ap.add_argument("--conf", type=float, default=0.05, help="ngưỡng thấp cho AP (model conf)")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)

    ov = {}
    if a.mode: ov["helmet_mode"] = a.mode
    if a.device: ov["device"] = a.device
    cfg = load_config(a.config, ov)
    datasets = load_datasets(a.datasets_file, only=a.datasets, max_images=a.max_images)
    if not datasets:
        raise SystemExit("Không có dataset nào sẵn — chạy scripts/prepare_datasets.py trước.")
    out = Path(a.out or ROOT / "outputs" / "eval" / time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"conf": a.conf}
    if a.imgsz: overrides["imgsz"] = a.imgsz
    results: list[EvalResult] = []
    for name in a.helmet:
        try:
            results += run_eval(cfg, name, datasets, a.stage, a.op_conf, overrides)
        except Exception as exc:  # noqa: BLE001
            log.exception("Lỗi với %s: %s", name, exc)
    md = [EvalResult.header()] + [r.row() for r in results]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    print("\n".join(md))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
