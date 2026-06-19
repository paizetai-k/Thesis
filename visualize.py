import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

DATASETS = {
    "full": {
        "players": [
            ("Stephen Curry", "data/stephen_curry_shots.csv"),
            ("LeBron James",  "data/lebron_james_shots.csv"),
            ("Kevin Durant",  "data/kevin_durant_shots.csv"),
        ],
        "all_label": "All shots  (2012-13 → 2018-19)",
    },
}


def draw_court(ax):
    lw = 1.5
    col = "black"

    hoop = plt.Circle((0, 0), 7.5, color=col, fill=False, linewidth=lw)
    ax.add_patch(hoop)

    ax.plot([-30, 30], [-7.5, -7.5], color=col, linewidth=lw)

    paint = patches.Rectangle((-80, -47.5), 160, 190,
                               fill=False, color=col, linewidth=lw)
    ax.add_patch(paint)

    inner = patches.Rectangle((-60, -47.5), 120, 190,
                               fill=False, color=col, linewidth=lw * 0.6,
                               linestyle="--")
    ax.add_patch(inner)

    ft_circle = patches.Arc((0, 142.5), 120, 120,
                             theta1=0, theta2=180,
                             color=col, linewidth=lw)
    ax.add_patch(ft_circle)

    ra = patches.Arc((0, 0), 80, 80,
                     theta1=0, theta2=180,
                     color=col, linewidth=lw)
    ax.add_patch(ra)

    ax.plot([-220, -220], [-47.5, 92.5], color=col, linewidth=lw)
    ax.plot([ 220,  220], [-47.5, 92.5], color=col, linewidth=lw)
    three_arc = patches.Arc((0, 0), 475, 475,
                             theta1=22, theta2=158,
                             color=col, linewidth=lw)
    ax.add_patch(three_arc)

    ax.plot([-250, 250], [422.5, 422.5], color=col, linewidth=lw)
    ax.plot([-250, 250], [-47.5, -47.5], color=col, linewidth=lw)
    ax.plot([-250, -250], [-47.5, 422.5], color=col, linewidth=lw)
    ax.plot([ 250,  250], [-47.5, 422.5], color=col, linewidth=lw)


def shot_chart(ax, df, title):
    missed = df[df["Shot Made Flag"] == 0]
    made   = df[df["Shot Made Flag"] == 1]

    ax.scatter(missed["X Location"], missed["Y Location"],
               c="red",  alpha=0.25, s=8, linewidths=0, label="Miss")
    ax.scatter(made["X Location"],   made["Y Location"],
               c="blue", alpha=0.35, s=8, linewidths=0, label="Make")

    draw_court(ax)

    fg = df["Shot Made Flag"].mean() * 100
    ax.set_title(f"{title}\nn={len(df):,}  FG%={fg:.1f}%", fontsize=10)
    ax.set_xlim(-260, 260)
    ax.set_ylim(-55, 430)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    for dataset, cfg in DATASETS.items():
        results_dir = f"results/{dataset}"
        os.makedirs(results_dir, exist_ok=True)

        for name, path in cfg["players"]:
            df   = pd.read_csv(path)
            train = df[df["Split"] == "train"]
            test  = df[df["Split"] == "test"]
            slug  = name.lower().replace(" ", "_")

            # Image 1: train | test split
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(name, fontsize=14, fontweight="bold")
            shot_chart(axes[0], train, "Train  (75%)")
            shot_chart(axes[1], test,  "Test   (25%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=10, markerscale=2, frameon=False)
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            out1 = f"{results_dir}/shot_chart_{slug}_split.png"
            fig.savefig(out1, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out1}")

            # Image 2: all shots combined
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.suptitle(name, fontsize=14, fontweight="bold")
            shot_chart(ax, df, cfg["all_label"])
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=10, markerscale=2, frameon=False)
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            out2 = f"{results_dir}/shot_chart_{slug}_all.png"
            fig.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out2}")


if __name__ == "__main__":
    main()
