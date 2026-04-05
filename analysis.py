"""
Analisis Data Likert - Model Adoption Customer Segmentation
Pendekatan: Technology Acceptance Model (TAM)

Variabel:
  PU   = Perceived Usefulness
  PEOU = Perceived Ease of Use
  ATU  = Attitude Toward Using
  BI   = Behavioral Intention
  ASU  = Actual System Use

Hipotesis TAM:
  H1: PEOU → PU
  H2: PEOU → ATU
  H3: PU   → ATU
  H4: PU   → BI
  H5: ATU  → BI
  H6: BI   → ASU
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os

os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/survey_likert.csv")

DEMO_COLS = ["Jenis_Kelamin", "Kelompok_Usia", "Pendidikan",
             "Pengalaman_Kerja", "Divisi"]

CONSTRUCTS = {
    "PU":   [c for c in df.columns if c.startswith("PU")],
    "PEOU": [c for c in df.columns if c.startswith("PEOU")],
    "ATU":  [c for c in df.columns if c.startswith("ATU")],
    "BI":   [c for c in df.columns if c.startswith("BI")],
    "ASU":  [c for c in df.columns if c.startswith("ASU")],
}

CONSTRUCT_NAMES = {
    "PU":   "Perceived Usefulness",
    "PEOU": "Perceived Ease of Use",
    "ATU":  "Attitude Toward Using",
    "BI":   "Behavioral Intention",
    "ASU":  "Actual System Use",
}

LIKERT_LABELS = {1: "Sangat Tidak Setuju", 2: "Tidak Setuju",
                 3: "Netral", 4: "Setuju", 5: "Sangat Setuju"}
COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]

print("=" * 65)
print("  ANALISIS DATA LIKERT — MODEL ADOPTION CUSTOMER SEGMENTATION")
print("=" * 65)
print(f"  Jumlah responden : {len(df)}")
print(f"  Total indikator  : {sum(len(v) for v in CONSTRUCTS.values())}")
print()


# ─────────────────────────────────────────────
# 1. STATISTIK DESKRIPTIF DEMOGRAFIS
# ─────────────────────────────────────────────
print("─" * 65)
print("1. STATISTIK DESKRIPTIF DEMOGRAFIS")
print("─" * 65)

for col in DEMO_COLS:
    freq = df[col].value_counts()
    pct = (freq / len(df) * 100).round(1)
    tbl = pd.DataFrame({"Frekuensi": freq, "Persentase (%)": pct})
    print(f"\n{col}:\n{tbl.to_string()}")

# ─────────────────────────────────────────────
# 2. UJI VALIDITAS (Pearson Correlation)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("2. UJI VALIDITAS (Korelasi Item-Total Terkoreksi)")
print("─" * 65)
print(f"  Kriteria valid: r_hitung > r_tabel (α=0.05, n={len(df)})")

# r_tabel untuk df = n-2
from scipy.stats import t as t_dist
alpha = 0.05
df_t = len(df) - 2
t_crit = t_dist.ppf(1 - alpha / 2, df_t)
r_tabel = t_crit / np.sqrt(df_t + t_crit**2)
print(f"  r_tabel = {r_tabel:.4f}\n")

validity_results = []
for construct, items in CONSTRUCTS.items():
    subset = df[items]
    total = subset.sum(axis=1)
    for item in items:
        # corrected item-total: korelasi item dengan total minus item itu sendiri
        corrected_total = total - subset[item]
        r, p = stats.pearsonr(subset[item], corrected_total)
        valid = "Valid ✓" if r > r_tabel else "Tidak Valid ✗"
        validity_results.append({
            "Konstruk": construct,
            "Indikator": item.split("_")[0],
            "r_hitung": round(r, 4),
            "r_tabel": round(r_tabel, 4),
            "p-value": round(p, 4),
            "Keterangan": valid,
        })

val_df = pd.DataFrame(validity_results)
print(val_df.to_string(index=False))

# ─────────────────────────────────────────────
# 3. UJI RELIABILITAS (Cronbach's Alpha)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("3. UJI RELIABILITAS (Cronbach's Alpha)")
print("─" * 65)
print("  Kriteria reliabel: α ≥ 0.70\n")

def cronbach_alpha(data):
    k = data.shape[1]
    item_var = data.var(axis=0, ddof=1).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var / total_var)

reliability_results = []
for construct, items in CONSTRUCTS.items():
    alpha_val = cronbach_alpha(df[items])
    status = "Reliabel ✓" if alpha_val >= 0.70 else "Tidak Reliabel ✗"
    reliability_results.append({
        "Konstruk": CONSTRUCT_NAMES[construct],
        "Kode": construct,
        "Jumlah Item": len(items),
        "Cronbach's α": round(alpha_val, 4),
        "Keterangan": status,
    })

rel_df = pd.DataFrame(reliability_results)
print(rel_df.to_string(index=False))

# ─────────────────────────────────────────────
# 4. STATISTIK DESKRIPTIF KONSTRUK
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("4. STATISTIK DESKRIPTIF KONSTRUK (Skor Rata-rata per Indikator)")
print("─" * 65)

all_desc = []
for construct, items in CONSTRUCTS.items():
    for item in items:
        col = df[item]
        all_desc.append({
            "Konstruk": construct,
            "Indikator": item.split("_")[0],
            "Min": col.min(),
            "Max": col.max(),
            "Mean": round(col.mean(), 3),
            "Std": round(col.std(), 3),
            "Median": col.median(),
        })

desc_df = pd.DataFrame(all_desc)
print(desc_df.to_string(index=False))

# Skor rata-rata per konstruk
print("\n  Skor Rata-rata Per Konstruk:")
for construct, items in CONSTRUCTS.items():
    m = df[items].mean().mean()
    std = df[items].stack().std()
    label = ""
    if m < 2.5:
        label = "Rendah"
    elif m < 3.5:
        label = "Sedang"
    else:
        label = "Tinggi"
    print(f"    {construct} ({CONSTRUCT_NAMES[construct]}): {m:.3f} ± {std:.3f} → {label}")

# ─────────────────────────────────────────────
# 5. UJI NORMALITAS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("5. UJI NORMALITAS (Kolmogorov-Smirnov)")
print("─" * 65)

# Hitung skor komposit
composite = {}
for construct, items in CONSTRUCTS.items():
    composite[construct] = df[items].mean(axis=1)
composite_df = pd.DataFrame(composite)

for col in composite_df.columns:
    stat, p = stats.kstest(
        composite_df[col],
        "norm",
        args=(composite_df[col].mean(), composite_df[col].std())
    )
    normal = "Normal ✓" if p > 0.05 else "Tidak Normal"
    print(f"  {col}: stat={stat:.4f}, p={p:.4f} → {normal}")

# ─────────────────────────────────────────────
# 6. KORELASI ANTAR KONSTRUK
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("6. KORELASI ANTAR KONSTRUK (Pearson)")
print("─" * 65)

corr_matrix = composite_df.corr()
print(corr_matrix.round(4).to_string())

# ─────────────────────────────────────────────
# 7. ANALISIS REGRESI TAM
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("7. ANALISIS REGRESI TAM")
print("─" * 65)

reg_df = composite_df.copy()

hypotheses = [
    ("H1", "PU",  "PEOU",       "PEOU → PU"),
    ("H2", "ATU", "PEOU",       "PEOU → ATU"),
    ("H3", "ATU", "PU",         "PU → ATU"),
    ("H4", "BI",  "PU",         "PU → BI"),
    ("H5", "BI",  "ATU",        "ATU → BI"),
    ("H6", "ASU", "BI",         "BI → ASU"),
]

# Multi-predictor models
models = [
    ("PU",  ["PEOU"],        "PEOU → PU"),
    ("ATU", ["PEOU", "PU"],  "PEOU + PU → ATU"),
    ("BI",  ["PU", "ATU"],   "PU + ATU → BI"),
    ("ASU", ["BI"],          "BI → ASU"),
]

print("\n  [ Model Regresi Lengkap TAM ]\n")
regression_summary = []
for dep, preds, label in models:
    X = sm.add_constant(reg_df[preds])
    y = reg_df[dep]
    model = sm.OLS(y, X).fit()
    print(f"  ── {label} ──")
    print(f"  R²={model.rsquared:.4f}  Adj-R²={model.rsquared_adj:.4f}"
          f"  F={model.fvalue:.3f}  p={model.f_pvalue:.4f}")
    for var in preds:
        coef = model.params[var]
        pval = model.pvalues[var]
        sig = "Signifikan ✓" if pval < 0.05 else "Tidak Signifikan"
        print(f"    β_{var}={coef:.4f}  p={pval:.4f}  → {sig}")
        regression_summary.append({
            "Model": label, "Prediktor": var,
            "β": round(coef, 4), "p-value": round(pval, 4),
            "Signifikan": sig,
        })
    print()

reg_sum_df = pd.DataFrame(regression_summary)

# ─────────────────────────────────────────────
# 8. VISUALISASI
# ─────────────────────────────────────────────
print("─" * 65)
print("8. MEMBUAT VISUALISASI...")
print("─" * 65)

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
plt.rcParams["font.family"] = "DejaVu Sans"

# --- 8a. Distribusi Likert per Konstruk ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribusi Respons Likert per Konstruk\nAdopsi Model Customer Segmentation",
             fontsize=14, fontweight="bold", y=1.01)

all_items = [(c, items) for c, items in CONSTRUCTS.items()]
for idx, (construct, items) in enumerate(all_items):
    ax = axes[idx // 3][idx % 3]
    stacked = pd.DataFrame(
        {item.split("_")[0]: df[item].value_counts().sort_index()
         for item in items}
    ).T.fillna(0)
    stacked.columns = [LIKERT_LABELS[i] for i in range(1, 6)]
    stacked.plot(kind="barh", stacked=True, ax=ax, color=COLORS,
                 edgecolor="white", linewidth=0.5)
    ax.set_title(f"{construct} — {CONSTRUCT_NAMES[construct]}", fontweight="bold")
    ax.set_xlabel("Jumlah Responden")
    ax.legend(loc="lower right", fontsize=7)
    ax.invert_yaxis()

axes[1][2].set_visible(False)
plt.tight_layout()
plt.savefig("output/01_distribusi_likert.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/01_distribusi_likert.png")

# --- 8b. Heatmap Korelasi Konstruk ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".3f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    xticklabels=[CONSTRUCT_NAMES[c] for c in corr_matrix.columns],
    yticklabels=[CONSTRUCT_NAMES[c] for c in corr_matrix.index],
)
ax.set_title("Heatmap Korelasi Antar Konstruk TAM\nAdopsi Model Customer Segmentation",
             fontweight="bold")
plt.tight_layout()
plt.savefig("output/02_heatmap_korelasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/02_heatmap_korelasi.png")

# --- 8c. Boxplot Skor Komposit ---
fig, ax = plt.subplots(figsize=(9, 5))
composite_plot = composite_df.rename(columns=CONSTRUCT_NAMES)
composite_plot.boxplot(ax=ax, patch_artist=True,
                       boxprops=dict(facecolor="#91bfdb", color="navy"),
                       medianprops=dict(color="red", linewidth=2),
                       whiskerprops=dict(color="navy"),
                       capprops=dict(color="navy"),
                       flierprops=dict(marker="o", color="gray", alpha=0.5))
ax.axhline(3, color="gray", linestyle="--", linewidth=1, label="Nilai Tengah (3)")
ax.set_ylim(1, 5.2)
ax.set_ylabel("Skor Rata-rata")
ax.set_title("Distribusi Skor Komposit Per Konstruk TAM", fontweight="bold")
ax.legend()
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("output/03_boxplot_komposit.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/03_boxplot_komposit.png")

# --- 8d. Diagram Path TAM ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("Diagram Path Technology Acceptance Model (TAM)\nAdopsi Model Customer Segmentation",
             fontsize=13, fontweight="bold")

# Node positions
nodes = {
    "PEOU": (1.5, 3),
    "PU":   (4.0, 4.5),
    "ATU":  (4.0, 1.5),
    "BI":   (7.0, 3),
    "ASU":  (9.2, 3),
}
node_colors = {"PEOU": "#4575b4", "PU": "#74add1",
               "ATU": "#fdae61", "BI": "#f46d43", "ASU": "#d73027"}

# Draw boxes
for name, (x, y) in nodes.items():
    fc = node_colors[name]
    rect = plt.Rectangle((x - 0.85, y - 0.45), 1.7, 0.9,
                          facecolor=fc, edgecolor="black",
                          linewidth=1.5, zorder=3, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x, y, f"{name}\n{CONSTRUCT_NAMES[name]}",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", zorder=4)

# Get regression betas for path labels
def get_beta(dep, pred):
    X = sm.add_constant(reg_df[[pred]])
    return sm.OLS(reg_df[dep], X).fit().params[pred]

edges = [
    ("PEOU", "PU"),
    ("PEOU", "ATU"),
    ("PU",   "ATU"),
    ("PU",   "BI"),
    ("ATU",  "BI"),
    ("BI",   "ASU"),
]

for (src, dst) in edges:
    x1, y1 = nodes[src]
    x2, y2 = nodes[dst]
    beta = get_beta(dst, src)
    color = "#2ca02c" if beta > 0 else "#d62728"
    ax.annotate(
        "", xy=(x2 - 0.85, y2), xytext=(x1 + 0.85, y1),
        arrowprops=dict(arrowstyle="->", color=color,
                        lw=1.5 + abs(beta) * 2)
    )
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.18, f"β={beta:.3f}", fontsize=8,
            color=color, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig("output/04_path_diagram_TAM.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/04_path_diagram_TAM.png")

# --- 8e. Demografis: Distribusi Responden ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Profil Demografis Responden", fontsize=13, fontweight="bold")

for ax, col in zip(axes, ["Jenis_Kelamin", "Kelompok_Usia", "Divisi"]):
    counts = df[col].value_counts()
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(counts)),
                edgecolor="white")
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("output/05_demografis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/05_demografis.png")

# --- 8f. Mean Skor Indikator per Konstruk (Radar-style bar) ---
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Rata-rata Skor per Indikator", fontsize=13, fontweight="bold")

for ax, (construct, items) in zip(axes, CONSTRUCTS.items()):
    means = df[items].mean()
    labels = [i.split("_")[0] for i in items]
    bars = ax.barh(labels, means, color=sns.color_palette("Blues_d", len(items)),
                   edgecolor="white")
    ax.set_xlim(1, 5)
    ax.axvline(3, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{construct}\n{CONSTRUCT_NAMES[construct]}", fontweight="bold")
    ax.set_xlabel("Mean (1–5)")
    for bar, val in zip(bars, means):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("output/06_mean_indikator.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Tersimpan: output/06_mean_indikator.png")

# ─────────────────────────────────────────────
# 9. EKSPOR RINGKASAN KE EXCEL
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("9. EKSPOR RINGKASAN KE EXCEL")
print("─" * 65)

with pd.ExcelWriter("output/ringkasan_analisis.xlsx", engine="openpyxl") as writer:
    val_df.to_excel(writer, sheet_name="Uji Validitas", index=False)
    rel_df.to_excel(writer, sheet_name="Uji Reliabilitas", index=False)
    desc_df.to_excel(writer, sheet_name="Statistik Deskriptif", index=False)
    corr_matrix.to_excel(writer, sheet_name="Korelasi Konstruk")
    reg_sum_df.to_excel(writer, sheet_name="Regresi TAM", index=False)
    composite_df.describe().round(4).to_excel(writer, sheet_name="Skor Komposit")

print("  Tersimpan: output/ringkasan_analisis.xlsx")

# ─────────────────────────────────────────────
# 10. INTERPRETASI RINGKAS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RINGKASAN HASIL ANALISIS")
print("=" * 65)

valid_count = val_df["Keterangan"].str.contains("Valid ✓").sum()
total_items = len(val_df)
print(f"\n  Validitas  : {valid_count}/{total_items} item valid (r > {r_tabel:.3f})")

rel_ok = rel_df["Keterangan"].str.contains("Reliabel ✓").sum()
print(f"  Reliabilitas: {rel_ok}/5 konstruk reliabel (α ≥ 0.70)")

for construct, items in CONSTRUCTS.items():
    m = df[items].mean().mean()
    print(f"  {construct} Mean : {m:.3f}")

print()
print("  Signifikansi Jalur TAM (p < 0.05):")
for _, row in reg_sum_df.iterrows():
    status = "✓" if "Signifikan ✓" in row["Signifikan"] else "✗"
    print(f"    {status} {row['Model'].split('→')[0].strip()} → "
          f"{row['Model'].split('→')[1].strip() if '→' in row['Model'] else ''}: "
          f"β={row['β']}, p={row['p-value']}")

print()
print("  File output tersimpan di folder: output/")
print("=" * 65)
