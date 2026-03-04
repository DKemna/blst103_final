"""
HOLC Redlining and Income/Wages Correlation Analysis
=====================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Chicago and current income levels by community area.

Data sources:
- HOLC redlining zones: local geojson.json
- Per capita income & poverty: Chicago Data Portal (kn9c-c2s2)
  "Census Data - Selected socioeconomic indicators in Chicago, 2008-2012"
- Community area boundaries: Chicago Data Portal (igwz-8jzy)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

# HOLC grade color mapping
COLOR_MAP = {
    "A": "#76a865",  # Green  – "Best"
    "B": "#7cbbe3",  # Blue   – "Still Desirable"
    "C": "#ffff00",  # Yellow – "Definitely Declining"
    "D": "#d9534f",  # Red    – "Hazardous" (redlined)
}

# Numeric encoding for correlation (higher = worse grade)
GRADE_NUM = {"A": 1, "B": 2, "C": 3, "D": 4}


# ─── 1. Load HOLC redlining data ────────────────────────────────────────────
print("Loading HOLC redlining data...")
holc = gpd.read_file(os.path.join(script_dir, "geojson.json"))

# Keep only residential zones with a valid grade (A-D)
holc = holc[holc["grade"].isin(["A", "B", "C", "D"])].copy()
holc = holc.to_crs(epsg=4326)  # ensure WGS84

print(f"  Loaded {len(holc)} HOLC zones")
for g in ["A", "B", "C", "D"]:
    print(f"    Grade {g}: {(holc['grade'] == g).sum()} zones")


# ─── 2. Download community area boundaries ──────────────────────────────────
print("\nDownloading Chicago community area boundaries...")
comm_url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson?$limit=100"
comm_areas = gpd.read_file(comm_url)
comm_areas = comm_areas.to_crs(epsg=4326)
comm_areas["area_numbe"] = comm_areas["area_numbe"].astype(int)
print(f"  Downloaded {len(comm_areas)} community areas")


# ─── 3. Download income / socioeconomic data ────────────────────────────────
print("\nDownloading income and socioeconomic data...")
income_url = (
    "https://data.cityofchicago.org/resource/kn9c-c2s2.json?$limit=100"
)
income_resp = requests.get(income_url, timeout=30)
income_data = pd.DataFrame(income_resp.json())

# Parse columns
income_data["ca"] = pd.to_numeric(
    income_data["ca"], errors="coerce"
)
# Dataset uses "community_area_number" or "ca" depending on version
if "ca" not in income_data.columns or income_data["ca"].isna().all():
    # Try alternative column name
    for col in income_data.columns:
        if "community_area_number" in col.lower() or col == "ca":
            income_data["ca"] = pd.to_numeric(income_data[col], errors="coerce")
            break

# Per capita income
pci_col = None
for col in income_data.columns:
    if "per_capita" in col.lower() and "income" in col.lower():
        pci_col = col
        break
if pci_col is None:
    # Fallback: look for any income column
    for col in income_data.columns:
        if "income" in col.lower():
            pci_col = col
            break
if pci_col is None:
    print(f"ERROR: Could not find per capita income column.")
    print(f"  Available columns: {income_data.columns.tolist()}")
    exit(1)

income_data["per_capita_income"] = pd.to_numeric(
    income_data[pci_col], errors="coerce"
)

# Poverty rate
pov_col = None
for col in income_data.columns:
    if "poverty" in col.lower():
        pov_col = col
        break
if pov_col:
    income_data["poverty_rate"] = pd.to_numeric(
        income_data[pov_col], errors="coerce"
    )
else:
    income_data["poverty_rate"] = np.nan

# Hardship index
hardship_col = None
for col in income_data.columns:
    if "hardship" in col.lower():
        hardship_col = col
        break
if hardship_col:
    income_data["hardship_index"] = pd.to_numeric(
        income_data[hardship_col], errors="coerce"
    )
else:
    income_data["hardship_index"] = np.nan

# Community area name
name_col = None
for col in income_data.columns:
    if "community_area_name" in col.lower() or col == "community_area_name":
        name_col = col
        break

income_clean = income_data[["ca", "per_capita_income"]].copy()
if pov_col:
    income_clean["poverty_rate"] = income_data["poverty_rate"]
if hardship_col:
    income_clean["hardship_index"] = income_data["hardship_index"]
if name_col:
    income_clean["community_area_name"] = income_data[name_col]

income_clean = income_clean.dropna(subset=["ca", "per_capita_income"])
income_clean["ca"] = income_clean["ca"].astype(int)

# Remove aggregate rows (e.g., city-wide row often has ca=0)
income_clean = income_clean[income_clean["ca"] > 0]

print(f"  Downloaded income data for {len(income_clean)} community areas")
print(
    f"  Per capita income range: "
    f"${income_clean['per_capita_income'].min():,.0f} – "
    f"${income_clean['per_capita_income'].max():,.0f}"
)
if "poverty_rate" in income_clean.columns:
    print(
        f"  Poverty rate range: "
        f"{income_clean['poverty_rate'].min():.1f}% – "
        f"{income_clean['poverty_rate'].max():.1f}%"
    )


# ─── 4. Merge community areas with income data ──────────────────────────────
print("\nMerging community areas with income data...")
comm_areas = comm_areas.merge(
    income_clean, left_on="area_numbe", right_on="ca", how="left"
)
comm_areas = comm_areas.dropna(subset=["per_capita_income"])


# ─── 5. Spatial overlay: find dominant HOLC grade per community area ─────────
print("\nCalculating dominant HOLC grade per community area...")

# Use a projected CRS for accurate area calculations
holc_proj = holc.to_crs(epsg=3857)
comm_proj = comm_areas.to_crs(epsg=3857)

# Overlay: intersect community areas with HOLC zones
overlay = gpd.overlay(comm_proj, holc_proj, how="intersection")
overlay["overlap_area"] = overlay.geometry.area

# For each community area, calculate area-weighted HOLC grade
results = []
for area_num in comm_areas["area_numbe"].unique():
    area_overlaps = overlay[overlay["area_numbe"] == area_num]
    if len(area_overlaps) == 0:
        continue

    # Total overlap area by grade
    grade_areas = area_overlaps.groupby("grade")["overlap_area"].sum()
    total_overlap = grade_areas.sum()

    if total_overlap == 0:
        continue

    # Dominant grade (largest area)
    dominant_grade = grade_areas.idxmax()

    # Area-weighted average grade score
    weighted_score = sum(
        GRADE_NUM[g] * (a / total_overlap) for g, a in grade_areas.items()
    )

    # Fraction of each grade
    grade_fracs = {
        g: grade_areas.get(g, 0) / total_overlap for g in ["A", "B", "C", "D"]
    }

    # Fraction that was redlined (D)
    redlined_frac = grade_fracs["D"]

    results.append(
        {
            "area_numbe": area_num,
            "dominant_grade": dominant_grade,
            "weighted_grade_score": weighted_score,
            "redlined_fraction": redlined_frac,
            "grade_A_frac": grade_fracs["A"],
            "grade_B_frac": grade_fracs["B"],
            "grade_C_frac": grade_fracs["C"],
            "grade_D_frac": grade_fracs["D"],
            "total_holc_overlap": total_overlap,
        }
    )

results_df = pd.DataFrame(results)
print(f"  Matched {len(results_df)} community areas with HOLC data")

# Merge with income data
analysis = comm_areas.merge(results_df, on="area_numbe", how="inner")
analysis = analysis.dropna(subset=["per_capita_income", "weighted_grade_score"])
print(f"  Final analysis dataset: {len(analysis)} community areas")


# ─── 6. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL RESULTS — HOLC Grade vs Income / Wages")
print("=" * 70)

# --- Pearson correlation: weighted grade score vs per capita income ---
r_weighted, p_weighted = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["per_capita_income"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs per capita income):")
print(f"    r = {r_weighted:.4f},  p = {p_weighted:.6f}")
if p_weighted < 0.05:
    print(f"    → Statistically significant (p < 0.05)")
if r_weighted < 0:
    print(
        f"    → Negative: worse HOLC grades associated with LOWER income"
    )
else:
    print(
        f"    → Positive: worse HOLC grades associated with higher income"
    )

# --- Spearman rank correlation ---
r_spear, p_spear = stats.spearmanr(
    analysis["weighted_grade_score"], analysis["per_capita_income"]
)
print(f"\n  Spearman rank correlation:")
print(f"    rₛ = {r_spear:.4f},  p = {p_spear:.6f}")

# --- Correlation: redlined fraction vs per capita income ---
r_redlined, p_redlined = stats.pearsonr(
    analysis["redlined_fraction"], analysis["per_capita_income"]
)
print(f"\n  Pearson correlation (redlined fraction vs per capita income):")
print(f"    r = {r_redlined:.4f},  p = {p_redlined:.6f}")
if p_redlined < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# --- Poverty rate correlation (if available) ---
if "poverty_rate" in analysis.columns and analysis["poverty_rate"].notna().sum() > 5:
    r_pov, p_pov = stats.pearsonr(
        analysis["weighted_grade_score"],
        analysis["poverty_rate"].dropna()
        if analysis["poverty_rate"].notna().all()
        else analysis.dropna(subset=["poverty_rate"])["weighted_grade_score"],
    )
    # Recompute properly
    pov_analysis = analysis.dropna(subset=["poverty_rate"])
    r_pov, p_pov = stats.pearsonr(
        pov_analysis["weighted_grade_score"], pov_analysis["poverty_rate"]
    )
    print(f"\n  Pearson correlation (weighted HOLC score vs poverty rate):")
    print(f"    r = {r_pov:.4f},  p = {p_pov:.6f}")
    if p_pov < 0.05:
        print(f"    → Statistically significant (p < 0.05)")
    if r_pov > 0:
        print(
            f"    → Positive: worse HOLC grades associated with HIGHER poverty"
        )

# --- Hardship index correlation (if available) ---
if "hardship_index" in analysis.columns and analysis["hardship_index"].notna().sum() > 5:
    hi_analysis = analysis.dropna(subset=["hardship_index"])
    r_hi, p_hi = stats.pearsonr(
        hi_analysis["weighted_grade_score"], hi_analysis["hardship_index"]
    )
    print(f"\n  Pearson correlation (weighted HOLC score vs hardship index):")
    print(f"    r = {r_hi:.4f},  p = {p_hi:.6f}")
    if p_hi < 0.05:
        print(f"    → Statistically significant (p < 0.05)")
    if r_hi > 0:
        print(
            f"    → Positive: worse HOLC grades associated with HIGHER hardship"
        )

# --- Mean income by dominant HOLC grade ---
print(f"\n  Mean per capita income by dominant HOLC grade:")
print(f"  {'Grade':<8} {'Mean $':<12} {'Median $':<12} {'Count':<6}")
print(f"  {'-' * 38}")
for grade in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == grade]
    if len(subset) > 0:
        mean_inc = subset["per_capita_income"].mean()
        median_inc = subset["per_capita_income"].median()
        print(
            f"  {grade:<8} ${mean_inc:<11,.0f} ${median_inc:<11,.0f} {len(subset):<6}"
        )

# --- One-way ANOVA ---
groups = [
    analysis[analysis["dominant_grade"] == g]["per_capita_income"].values
    for g in ["A", "B", "C", "D"]
    if len(analysis[analysis["dominant_grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (per capita income across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    → Significant difference between groups (p < 0.05)")

# --- Point-biserial: redlined (D) vs not ---
analysis["is_redlined"] = (analysis["dominant_grade"] == "D").astype(int)
r_pb, p_pb = stats.pointbiserialr(
    analysis["is_redlined"], analysis["per_capita_income"]
)
print(f"\n  Point-biserial correlation (redlined D vs other grades):")
print(f"    r_pb = {r_pb:.4f},  p = {p_pb:.6f}")
if p_pb < 0.05:
    print(
        f"    → Significant income difference between redlined and non-redlined areas"
    )

# --- Effect size: Cohen's d (Grade A vs Grade D) ---
a_income = analysis[analysis["dominant_grade"] == "A"]["per_capita_income"]
d_income = analysis[analysis["dominant_grade"] == "D"]["per_capita_income"]
if len(a_income) > 1 and len(d_income) > 1:
    pooled_std = np.sqrt(
        (
            (len(a_income) - 1) * a_income.std() ** 2
            + (len(d_income) - 1) * d_income.std() ** 2
        )
        / (len(a_income) + len(d_income) - 2)
    )
    if pooled_std > 0:
        cohens_d = (a_income.mean() - d_income.mean()) / pooled_std
        print(f"\n  Cohen's d (Grade A vs Grade D):")
        print(f"    d = {cohens_d:.4f}")
        if abs(cohens_d) >= 0.8:
            print(f"    → Large effect size")
        elif abs(cohens_d) >= 0.5:
            print(f"    → Medium effect size")
        elif abs(cohens_d) >= 0.2:
            print(f"    → Small effect size")

print("=" * 70)


# ─── 7. Visualization ───────────────────────────────────────────────────────
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle(
    "HOLC Redlining (1930s) and Current Income Levels in Chicago",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# --- Panel 1: Scatter – weighted grade score vs per capita income ---
ax1 = axes[0, 0]
colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax1.scatter(
    analysis["weighted_grade_score"],
    analysis["per_capita_income"] / 1000,
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)

# Add regression line
slope, intercept, r, p, se = stats.linregress(
    analysis["weighted_grade_score"], analysis["per_capita_income"] / 1000
)
x_line = np.linspace(1, 4, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax1.set_ylabel("Per Capita Income ($K)", fontsize=11)
ax1.set_title(
    f"Weighted HOLC Score vs Per Capita Income\nr={r:.3f}, p={p:.4f}",
    fontsize=13,
)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax1.grid(True, alpha=0.3)

# --- Panel 2: Box plot – income by dominant grade ---
ax2 = axes[0, 1]
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == g]["per_capita_income"]
    if len(subset) > 0:
        grade_data.append(subset.values / 1000)
        grade_labels.append(f"Grade {g}\n(n={len(subset)})")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Per Capita Income ($K)", fontsize=11)
ax2.set_title("Income Distribution by Dominant HOLC Grade", fontsize=13)
ax2.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Scatter – weighted HOLC score vs poverty rate ---
ax3 = axes[1, 0]
pov_analysis = analysis.dropna(subset=["poverty_rate"])
pov_colors = [COLOR_MAP[g] for g in pov_analysis["dominant_grade"]]
ax3.scatter(
    pov_analysis["weighted_grade_score"],
    pov_analysis["poverty_rate"],
    c=pov_colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
# Regression line
slope3, intercept3, r3, p3, se3 = stats.linregress(
    pov_analysis["weighted_grade_score"], pov_analysis["poverty_rate"]
)
x_line3 = np.linspace(1, 4, 100)
ax3.plot(x_line3, slope3 * x_line3 + intercept3, "k--", linewidth=2, alpha=0.7)
ax3.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax3.set_ylabel("Poverty Rate (%)", fontsize=11)
ax3.set_title(
    f"Weighted HOLC Score vs Poverty Rate\nr={r3:.3f}, p={p3:.6f}",
    fontsize=13,
)
ax3.set_xticks([1, 2, 3, 4])
ax3.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax3.grid(True, alpha=0.3)

# Add annotation for significance
ax3.annotate(
    "Statistically significant (p < 0.05)",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    fontsize=9,
    fontweight="bold",
    color="darkred",
    va="top",
)

# --- Panel 4: Choropleth map – income with HOLC overlay ---
ax4 = axes[1, 1]
analysis_geo = analysis.copy()
analysis_geo = analysis_geo.to_crs(epsg=3857)
analysis_geo.plot(
    column="per_capita_income",
    cmap="RdYlGn",
    legend=True,
    ax=ax4,
    edgecolor="gray",
    linewidth=0.5,
    alpha=0.8,
    legend_kwds={
        "label": "Per Capita Income ($)",
        "shrink": 0.7,
        "format": "${x:,.0f}",
    },
)

# Overlay HOLC boundaries
holc_plot = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_plot[holc_plot["grade"] == grade]
    subset.boundary.plot(ax=ax4, color=COLOR_MAP[grade], linewidth=0.4, alpha=0.5)

ax4.set_axis_off()
ax4.set_title("Per Capita Income Map with HOLC Zone Boundaries", fontsize=13)

# Add legend for HOLC grades
legend_elements = [
    Patch(facecolor=c, edgecolor="black", label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(script_dir, "holc_income_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  Saved figure to {output_path}")
plt.show()

print("\nDone!")
