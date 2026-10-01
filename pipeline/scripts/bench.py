"""Benchmark MỌI candidate mũ (role helmet/openvocab) trên mọi dataset + đo latency GPU/CPU.

  python scripts/bench.py                       # detector full-frame
  python scripts/bench.py --stage pipeline --mode crop
  python scripts/bench.py --only modelA modelB --cpu
Ghi outputs/bench/<time>/results.md, results.json, latency.json và bảng xếp hạng.
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
from helmet_pipeline.runner import helmet_candidates, load_datasets, measure_latency, run_eval, sample_images  # noqa: E402

log = logging.getLogger("bench")


def rank(results: list[EvalResult]) -> list[dict]:
    """Điểm tổng hợp theo model: trung bình mAP50 và recall no_helmet@P90 qua các dataset."""
    by: dict[str, list[EvalResult]] = {}
    for r in results:
        by.setdefault(r.model, []).append(r)
    rows = []
    def _mean(vals):
        vals = [v for v in vals if v is not None and v == v]        # bỏ NaN (dataset không có lớp đó)
        return sum(vals) / len(vals) if vals else float("nan")
    for m, rs in by.items():
        map50 = _mean(r.map50 for r in rs)
        r90 = _mean(r.recall_no_helmet_at_p90 for r in rs if r.n_gt.get("no_helmet", 0) > 0)
        ap_noh = _mean(r.ap50.get("no_helmet") for r in rs)
        score = _mean([map50, ap_noh])
        rows.append({"model": m, "datasets": len(rs), "mean_mAP50": round(map50, 4), "mean_AP50_no_helmet": round(ap_noh, 4),
                     "mean_R_no_helmet@P90": round(r90, 4), "score": round(score, 4)})
    return sorted(rows, key=lambda r: -r["score"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--only", nargs="*", help="chỉ các candidate này")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--stage", choices=["detector", "pipeline"], default="detector")
    ap.add_argument("--mode", choices=["full", "crop", "both"])
    ap.add_argument("--op-conf", type=float, default=0.4)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--cpu", action="store_true", help="đo thêm latency CPU")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)

    ov = {"helmet_mode": a.mode} if a.mode else {}
    cfg = load_config(a.config, ov)
    cands = helmet_candidates()
    if a.only:
        cands = {k: v for k, v in cands.items() if k in a.only}
    datasets = load_datasets(only=a.datasets, max_images=a.max_images)
    if not datasets:
        raise SystemExit("Không có dataset nào sẵn — chạy scripts/prepare_datasets.py trước.")
    out = Path(a.out or ROOT / "outputs" / "bench" / time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    log.info("Benchmark %d model x %d dataset -> %s", len(cands), len(datasets), out)

    results: list[EvalResult] = []
    lat: list[dict] = []
    imgs = sample_images(datasets, 8)
    for name in cands:
        t0 = time.perf_counter()
        try:
            results += run_eval(cfg, name, datasets, a.stage, a.op_conf, {"conf": a.conf})
            if not a.no_latency and imgs:
                lat.append(measure_latency(cfg, name, imgs, None, a.stage))
                if a.cpu:
                    lat.append(measure_latency(cfg, name, imgs, "cpu", a.stage, runs=10))
        except Exception as exc:  # noqa: BLE001
            log.exception("Lỗi với %s: %s", name, exc)
            lat.append({"model": name, "error": str(exc)})
        log.info("%s xong sau %.0fs", name, time.perf_counter() - t0)
        # ghi dần để không mất kết quả nếu dừng giữa chừng
        (out / "results.md").write_text("\n".join([EvalResult.header()] + [r.row() for r in results]), encoding="utf-8")
        json.dump([r.to_dict() for r in results], open(out / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(lat, open(out / "latency.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ranking = rank(results)
    lines = ["# Bảng xếp hạng", "", "| # | model | datasets | mean mAP50 | mean AP50 no_helmet | mean R no_helmet@P90 | score |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranking, 1):
        lines.append(f"| {i} | {r['model']} | {r['datasets']} | {r['mean_mAP50']:.3f} | {r['mean_AP50_no_helmet']:.3f} | {r['mean_R_no_helmet@P90']:.3f} | {r['score']:.3f} |")
    lines += ["", "# Chi tiết", "", EvalResult.header()] + [r.row() for r in results]
    if lat:
        lines += ["", "# Latency (ảnh thật, batch 1)", "", "| model | device | stage | mean ms | p50 | p90 | FPS |", "|---|---|---|---|---|---|---|"]
        for l in lat:
            if "error" in l:
                lines.append(f"| {l['model']} | - | - | lỗi: {l['error'][:60]} | | | |")
            else:
                lines.append(f"| {l['model']} | {l['device']} | {l['stage']} | {l['mean_ms']:.1f} | {l['p50_ms']:.1f} | {l['p90_ms']:.1f} | {l['fps']:.1f} |")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
