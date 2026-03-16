#!/usr/bin/env python3
"""
layer_repeat_experiment.json 결과로 DER/Spk_Count_Acc 히트맵 시각화.

사용법:
    python scripts/plot_layer_heatmap.py --input results/layer_repeat_realworld.json
    python scripts/plot_layer_heatmap.py --input results/layer_repeat_realworld.json --metric Spk_Count_Acc
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    return data


def build_heatmap(data: dict, metric: str = "DER"):
    num_layers = data["num_layers"]
    baseline_val = data["baseline"][metric]

    # (num_layers x num_layers) 행렬, NaN으로 초기화
    mat = np.full((num_layers, num_layers), np.nan)

    for r in data["results"]:
        if r["block_size"] == 0:
            continue
        i = r["start_layer"]
        j = r["end_layer"]
        val = r[metric]
        # delta: DER는 낮을수록 좋으므로 baseline - val (양수 = 개선)
        # Spk_Count_Acc는 높을수록 좋으므로 val - baseline
        if metric == "DER" or metric in ("CER", "FA", "MISS"):
            delta = baseline_val - val   # 양수 = 개선(DER 감소)
        else:
            delta = val - baseline_val   # 양수 = 개선(Acc 증가)
        mat[i, j] = delta

    return mat, baseline_val


def plot_heatmap(data: dict, metric: str, out_path: str):
    num_layers = data["num_layers"]
    mat, baseline_val = build_heatmap(data, metric)

    # 색상 범위 대칭
    vmax = np.nanmax(np.abs(mat))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── 왼쪽: 히트맵 ──
    ax = axes[0]
    im = ax.imshow(
        mat,
        origin="upper",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label=f"Delta {metric} (positive=improvement)")

    # 최적 지점 표시
    best_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    best_i, best_j = best_idx
    ax.scatter(best_j, best_i, s=200, facecolors="none",
               edgecolors="lime", linewidths=2.5, zorder=5,
               label=f"Best ({best_i},{best_j}): Δ={mat[best_i,best_j]:.4f}")

    # 블록 크기별 대각선 표시
    block_colors = {2: "white", 3: "yellow", 4: "cyan", 5: "orange"}
    for bs, color in block_colors.items():
        xs, ys = [], []
        for i in range(num_layers - bs + 1):
            j = i + bs - 1
            if j < num_layers:
                xs.append(j)
                ys.append(i)
        if xs:
            ax.plot(xs, ys, ".", color=color, markersize=3, alpha=0.4,
                    label=f"block_size={bs}")

    ax.set_xlabel("j (end layer)")
    ax.set_ylabel("i (start layer)")
    title_suffix = "↓ better" if metric in ("DER", "CER", "FA", "MISS") else "↑ better"
    ax.set_title(
        f"Layer Repeat Heatmap — {metric} delta ({title_suffix})\n"
        f"Baseline {metric}: {baseline_val:.4f} | Model: {Path(data['model']).stem}"
    )
    ax.set_xticks(range(num_layers))
    ax.set_yticks(range(num_layers))
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", fontsize=8)

    # ── 오른쪽: 상위 10 결과 표 ──
    ax2 = axes[1]
    ax2.axis("off")

    results_sorted = sorted(
        [r for r in data["results"] if r["block_size"] > 0],
        key=lambda r: (
            (data["baseline"][metric] - r[metric])
            if metric in ("DER", "CER", "FA", "MISS")
            else (r[metric] - data["baseline"][metric])
        ),
        reverse=True,
    )[:15]

    table_data = []
    for r in results_sorted:
        delta = (
            (data["baseline"][metric] - r[metric])
            if metric in ("DER", "CER", "FA", "MISS")
            else (r[metric] - data["baseline"][metric])
        )
        table_data.append([
            f"{r['start_layer']}~{r['end_layer']}",
            f"{r['block_size']}",
            f"{r[metric]:.4f}",
            f"{delta:+.4f}",
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Layers", "Block", metric, f"D{metric}"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # 개선된 행 하이라이트
    for row_idx, (row_data, r) in enumerate(zip(table_data, results_sorted)):
        delta = float(row_data[3])
        if delta > 0:
            for col in range(4):
                table[(row_idx + 1, col)].set_facecolor("#d4edda")

    ax2.set_title(f"Top 15 Configs ({metric})", pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close()


def plot_skyline(data: dict, metric: str, out_path: str):
    """레이어별 평균 delta (skyline plot)"""
    num_layers = data["num_layers"]
    mat, baseline_val = build_heatmap(data, metric)

    row_means = np.nanmean(mat, axis=1)   # start 레이어별 평균
    col_means = np.nanmean(mat, axis=0)   # end 레이어별 평균

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors_row = ["#d73027" if v > 0 else "#4575b4" for v in row_means]
    ax1.bar(range(num_layers), row_means, color=colors_row)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("i (start layer)")
    ax1.set_ylabel(f"Avg D{metric}")
    ax1.set_title(f"Avg delta by start layer\n(red=improvement, blue=degradation)")
    ax1.set_xticks(range(num_layers))

    colors_col = ["#d73027" if v > 0 else "#4575b4" for v in col_means]
    ax2.bar(range(num_layers), col_means, color=colors_col)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("j (end layer)")
    ax2.set_ylabel(f"Avg D{metric}")
    ax2.set_title(f"Avg delta by end layer\n(red=improvement, blue=degradation)")
    ax2.set_xticks(range(num_layers))

    plt.suptitle(
        f"Skyline Plot — {metric} | Baseline: {baseline_val:.4f}",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/layer_repeat_realworld.json")
    parser.add_argument(
        "--metric",
        default="DER",
        choices=["DER", "CER", "FA", "MISS", "Spk_Count_Acc"],
    )
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    data = load_results(args.input)
    dataset = data.get("dataset", "unknown")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 히트맵
    heatmap_path = str(out_dir / f"heatmap_{dataset}_{args.metric}.png")
    plot_heatmap(data, args.metric, heatmap_path)

    # 스카이라인
    skyline_path = str(out_dir / f"skyline_{dataset}_{args.metric}.png")
    plot_skyline(data, args.metric, skyline_path)

    # 추가로 Spk_Count_Acc도 함께 출력
    if args.metric == "DER":
        plot_heatmap(data, "Spk_Count_Acc",
                     str(out_dir / f"heatmap_{dataset}_Spk_Count_Acc.png"))
        plot_skyline(data, "Spk_Count_Acc",
                     str(out_dir / f"skyline_{dataset}_Spk_Count_Acc.png"))


if __name__ == "__main__":
    main()
