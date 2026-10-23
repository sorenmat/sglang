"""Charts for the plain-language host-KV results report (palette: cascade cold/minimal)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Cascade palette (pdf.py palette.cascade, cold/minimal)
HEADER = "#556a74"   # M tier - slate (vanilla / baseline)
ICON = "#47809c"     # S tier - steel blue (ours / tier)
ACCENT = "#c43e54"   # XS tier - highlight
BORDER = "#bbc6cb"
TEXT = "#1b1d1e"
MUTED = "#80868a"
SUCCESS = "#479561"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": TEXT,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})

OUT = "/home/smo/code/sglang-perf/benchmark/context_certificate/report_build"


def strip_axes(ax, keep_left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_color(BORDER)


# ── Chart 1: fair fight at 101k (vanilla slightly faster) ──────────────────
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ns = [1, 4, 8]
van = [22.7, 55.5, 113.6]
ours = [28.0, 60.4, 124.4]
x = np.arange(len(ns))
w = 0.34
b1 = ax.bar(x - w / 2, van, w, color=HEADER, edgecolor="none", label="Standard serving")
b2 = ax.bar(x + w / 2, ours, w, color=ICON, edgecolor="none", label="With the overflow shelf")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}s", (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=11, color=TEXT)
ax.set_xticks(x)
ax.set_xticklabels(["1 document\nat a time", "4 documents\nat once", "8 documents\nat once"])
ax.set_yticks([])
ax.set_ylim(0, 145)
strip_axes(ax)
ax.legend(frameon=False, loc="upper left", handlelength=1.0, handleheight=1.0)
fig.tight_layout(pad=1.5)
fig.savefig(f"{OUT}/c1_fair.png", dpi=200)
plt.close(fig)

# ── Chart 2: near the ceiling, ours wins ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 4.4))
vals = [42, 39]
labels = ["Standard serving", "With the overflow shelf"]
bars = ax.bar([0, 1], vals, 0.45, color=[HEADER, ICON], edgecolor="none")
for r, v in zip(bars, vals):
    ax.annotate(f"{v}s", (r.get_x() + r.get_width() / 2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=13, color=TEXT)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([])
ax.set_ylim(0, 60)
strip_axes(ax)
ax.annotate("7% faster when documents push\nthe desk right to its limit",
            xy=(1, 39), xytext=(0.62, 52), fontsize=11, color=ACCENT, ha="left")
fig.tight_layout(pad=1.5)
fig.savefig(f"{OUT}/c2_ceiling.png", dpi=200)
plt.close(fig)

# ── Chart 3: donut — where a big document lives during generation ──────────
fig, ax = plt.subplots(figsize=(7.4, 4.0))
sizes = [83, 17]
cols = [ICON, HEADER]
wedges, _ = ax.pie(sizes, colors=cols, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2))
ax.text(0, 0.08, "83%", ha="center", va="center", fontsize=30, color=TEXT, fontweight="bold")
ax.text(0, -0.24, "of the document\nrests on the shelf", ha="center", va="center",
        fontsize=10.5, color=MUTED)
ax.legend(wedges, ["Shelf — main memory (83%)", "Desk — the chip (17%)"],
          loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False,
          fontsize=12, handlelength=1.0, handleheight=1.0, labelspacing=1.0)
ax.set(aspect="equal")
fig.tight_layout(pad=0.8)
fig.savefig(f"{OUT}/c3_shelf.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Chart 4: our own tuning gains ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 4.2))
vals = [66.9, 51.8]
bars = ax.bar([0, 1], vals, 0.42, color=[BORDER, SUCCESS], edgecolor="none")
for r, v in zip(bars, vals):
    ax.annotate(f"{v}s", (r.get_x() + r.get_width() / 2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=13, color=TEXT)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Our first working version", "After tuning"], fontsize=12)
ax.set_yticks([])
ax.set_ylim(0, 82)
strip_axes(ax)
ax.annotate("22% faster on the toughest job\n(4 documents at once, worst case)",
            xy=(0.5, 76), fontsize=11, color=SUCCESS, ha="center")
fig.tight_layout(pad=1.5)
fig.savefig(f"{OUT}/c4_tuning.png", dpi=200)
plt.close(fig)

# ── Chart 5: soak sawtooth — the maintenance clock ──────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.2))
t, v = [0.0], [0.0]
clock = 0.0
for cycle in range(24):
    for step in range(1, 6):
        t.append(clock + step * 2.0)
        v.append(min(31.0, step * 6.2))
    t.append(clock + 10.0)
    v.append(0.0)
    clock += 10.0
ax.plot(t, v, color=ICON, linewidth=2.2, solid_joinstyle="round")
ax.fill_between(t, v, color=ICON, alpha=0.12, linewidth=0)
ax.set_xlim(0, 242)
ax.set_ylim(0, 40)
ax.set_xticks([0, 60, 120, 180, 235])
ax.set_xticklabels(["start", "1 hour", "2 hours", "3 hours", "4 hours"])
ax.set_yticks([])
strip_axes(ax, keep_left=False)
ax.spines["bottom"].set_color(BORDER)
ax.annotate("desk slowly fills up with the known leak...\n...a 2-minute service stop resets it (24 in total)",
            xy=(9.5, 30.5), xytext=(30, 35.2),
            fontsize=11, color=MUTED, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.text(236, 36.5, "0 crashes · 0 wrong answers", fontsize=11.5, color=SUCCESS,
        ha="right", fontweight="bold")
fig.tight_layout(pad=1.5)
fig.savefig(f"{OUT}/c5_soak.png", dpi=200)
plt.close(fig)

print("charts done")
