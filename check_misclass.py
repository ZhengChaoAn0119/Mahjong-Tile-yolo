"""
check_misclass.py
Confusion-matrix based misclassification analysis.
Shows which classes get confused with each other (top FP pairs),
plus per-class FP / FN counts.

Usage:
  python check_misclass.py [model_path]

Default model: runs/detect/majsoul_bgswap/weights/best.pt
Output: runs/detect/<run_name>/analysis/misclass_report.json
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
DATA_YAML    = PROJECT_ROOT / "dataset_merged/data.yaml"

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]
NC = len(CLASS_NAMES)


def run_analysis(model_path: Path):
    out_dir = model_path.parent.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model : {model_path}")
    model = YOLO(str(model_path))

    # Run val with save_json to get per-image predictions
    results = model.val(
        data    = str(DATA_YAML),
        imgsz   = 640,
        batch   = 16,
        device  = 0,
        split   = "val",
        verbose = False,
        plots   = True,   # saves confusion_matrix.png in run dir
    )

    # ── Per-class mAP50 ──────────────────────────────────────────────────────
    metrics   = results.box
    ap50      = metrics.ap50          # (NC,)
    precision = metrics.p
    recall    = metrics.r
    overall   = float(metrics.map50)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "class_id":  i,
            "precision": float(precision[i]) if i < len(precision) else 0.0,
            "recall":    float(recall[i])    if i < len(recall)    else 0.0,
            "mAP50":     float(ap50[i])      if i < len(ap50)      else 0.0,
        }

    # ── Confusion matrix ─────────────────────────────────────────────────────
    # ultralytics stores it in results.confusion_matrix
    cm_obj = results.confusion_matrix
    matrix = cm_obj.matrix   # shape: (NC+1, NC+1), last row/col = background

    # FP per class = sum of column i (predicted as i) minus TP
    # FN per class = sum of row i (actual i) minus TP
    fp_counts = {}
    fn_counts = {}
    confused_pairs = []

    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp   # others predicted as class i
        fn = matrix[i, :].sum() - tp   # class i predicted as others
        fp_counts[name] = int(fp)
        fn_counts[name] = int(fn)

    # Find top confused pairs (predicted_as != actual, excluding background)
    for actual in range(NC):
        for pred in range(NC):
            if actual == pred:
                continue
            count = int(matrix[actual, pred])
            if count > 0:
                confused_pairs.append({
                    "actual":     CLASS_NAMES[actual],
                    "predicted":  CLASS_NAMES[pred],
                    "count":      count,
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_confused = confused_pairs[:20]

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"MISCLASSIFICATION REPORT  (overall mAP50: {overall:.4f})")
    print("="*65)

    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]["mAP50"])
    print(f"\n{'Class':<10} {'mAP50':>7} {'FP':>5} {'FN':>5}")
    print("-"*30)
    for name, m in sorted_cls:
        flag = " ←" if m["mAP50"] < 0.90 else ""
        print(f"{name:<10} {m['mAP50']:>7.3f} {fp_counts[name]:>5} {fn_counts[name]:>5}{flag}")

    print(f"\nTop confused pairs (actual → predicted):")
    print(f"  {'Actual':<10} {'Predicted':<10} {'Count':>6}")
    print("  " + "-"*28)
    for p in top_confused[:15]:
        print(f"  {p['actual']:<10} {p['predicted']:<10} {p['count']:>6}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "overall_mAP50": overall,
        "per_class":     per_class,
        "fp_per_class":  fp_counts,
        "fn_per_class":  fn_counts,
        "top_confused_pairs": top_confused,
    }
    report_file = out_dir / "misclass_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_file}")

    plot_visuals(per_class, fp_counts, fn_counts, top_confused, matrix, overall, out_dir)

    return report


def plot_visuals(per_class, fp_counts, fn_counts, top_confused, matrix, overall, out_dir):
    # ── 1. Per-class mAP50 bar chart ────────────────────────────────────────
    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]["mAP50"])
    names  = [n for n, _ in sorted_cls]
    map50s = [m["mAP50"] for _, m in sorted_cls]
    colors = ["#e74c3c" if v < 0.85 else "#e67e22" if v < 0.90 else "#2ecc71" for v in map50s]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(names, map50s, color=colors)
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1.2, label="0.90 target")
    ax.axvline(overall, color="steelblue", linestyle=":", linewidth=1.2, label=f"overall {overall:.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("mAP50")
    ax.set_title(f"Per-class mAP50  (overall={overall:.4f})")
    red_p   = mpatches.Patch(color="#e74c3c", label="< 0.85")
    org_p   = mpatches.Patch(color="#e67e22", label="0.85–0.90")
    grn_p   = mpatches.Patch(color="#2ecc71", label="≥ 0.90")
    ax.legend(handles=[red_p, org_p, grn_p, ax.lines[0], ax.lines[1]], fontsize=8)
    for bar, val in zip(bars, map50s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    out1 = out_dir / "per_class_map50.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── 2. FP / FN grouped bar chart ────────────────────────────────────────
    # Show only classes with FP+FN > 0, sorted by total errors
    err_cls = [(n, fp_counts[n], fn_counts[n]) for n in CLASS_NAMES if fp_counts[n] + fn_counts[n] > 0]
    err_cls.sort(key=lambda x: x[1] + x[2], reverse=True)
    e_names = [x[0] for x in err_cls]
    fps     = [x[1] for x in err_cls]
    fns     = [x[2] for x in err_cls]

    x = np.arange(len(e_names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, fps, w, label="FP (false positive)", color="#e74c3c", alpha=0.85)
    ax.bar(x + w/2, fns, w, label="FN (false negative)", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(e_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("False Positive / False Negative per class")
    ax.legend()
    plt.tight_layout()
    out2 = out_dir / "fp_fn_per_class.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Saved: {out2}")

    # ── 3. Top confused pairs heatmap (sub-matrix) ──────────────────────────
    # Extract the NC×NC sub-matrix (exclude background row/col)
    sub = matrix[:NC, :NC].copy()
    np.fill_diagonal(sub, 0)   # hide TP diagonal so errors are visible

    # Only show classes that have at least one off-diagonal entry
    active = [i for i in range(NC) if sub[i].sum() > 0 or sub[:, i].sum() > 0]
    if active:
        sub_filt = sub[np.ix_(active, active)]
        act_names = [CLASS_NAMES[i] for i in active]

        fig, ax = plt.subplots(figsize=(max(8, len(active) * 0.55), max(7, len(active) * 0.5)))
        im = ax.imshow(sub_filt, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(act_names)))
        ax.set_yticks(range(len(act_names)))
        ax.set_xticklabels(act_names, rotation=90, fontsize=7)
        ax.set_yticklabels(act_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion sub-matrix (diagonal=0, actual→predicted)")
        plt.colorbar(im, ax=ax, fraction=0.03)
        for i in range(len(act_names)):
            for j in range(len(act_names)):
                v = int(sub_filt[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=7, color="black" if v < sub_filt.max() * 0.6 else "white")
        plt.tight_layout()
        out3 = out_dir / "confusion_submatrix.png"
        fig.savefig(out3, dpi=120)
        plt.close(fig)
        print(f"Saved: {out3}")


if __name__ == "__main__":
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                 PROJECT_ROOT / "runs/detect/majsoul_bgswap/weights/best.pt"
    run_analysis(model_path)
