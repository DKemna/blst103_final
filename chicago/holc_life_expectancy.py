"""
HOLC Redlining and Life Expectancy Correlation Analysis
========================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Chicago and current life expectancy by community area.

Data sources:
- HOLC redlining zones: local geojson.json
- Life expectancy: Chicago Data Portal (qjr3-bm53)
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


# ─── 3. Download life expectancy data ───────────────────────────────────────
print("\nDownloading life expectancy data...")
le_url = "https://data.cityofchicago.org/resource/qjr3-bm53.json?$limit=100"
le_resp = requests.get(le_url)
le_data = pd.DataFrame(le_resp.json())

# Use 2010 life expectancy (most recent in this dataset)
le_data["ca"] = pd.to_numeric(le_data["ca"], errors="coerce")
le_data["life_expectancy"] = pd.to_numeric(
    le_data["_2010_life_expectancy"], errors="coerce"
)
le_data = le_data[["ca", "community_area", "life_expectancy"]].dropna()
le_data["ca"] = le_data["ca"].astype(int)
print(f"  Downloaded life expectancy for {len(le_data)} community areas")
print(f"  Range: {le_data['life_expectancy'].min():.1f} – {le_data['life_expectancy'].max():.1f} years")


# ─── 4. Merge community areas with life expectancy ──────────────────────────
print("\nMerging community areas with life expectancy...")
comm_areas = comm_areas.merge(le_data, left_on="area_numbe", right_on="ca", how="left")
comm_areas = comm_areas.dropna(subset=["life_expectancy"])


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
    grade_fracs = {g: grade_areas.get(g, 0) / total_overlap for g in ["A", "B", "C", "D"]}

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

# Merge with life expectancy
analysis = comm_areas.merge(results_df, on="area_numbe", how="inner")
analysis = analysis.dropna(subset=["life_expectancy", "weighted_grade_score"])
print(f"  Final analysis dataset: {len(analysis)} community areas")


# ─── 6. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STATISTICAL RESULTS")
print("=" * 65)

# Correlation: weighted grade score vs life expectancy
r_weighted, p_weighted = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["life_expectancy"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs life expectancy):")
print(f"    r = {r_weighted:.4f},  p = {p_weighted:.6f}")
if p_weighted < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# Correlation: redlined fraction vs life expectancy
r_redlined, p_redlined = stats.pearsonr(
    analysis["redlined_fraction"], analysis["life_expectancy"]
)
print(f"\n  Pearson correlation (redlined fraction vs life expectancy):")
print(f"    r = {r_redlined:.4f},  p = {p_redlined:.6f}")
if p_redlined < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# Mean life expectancy by dominant HOLC grade
print(f"\n  Mean life expectancy by dominant HOLC grade:")
print(f"  {'Grade':<8} {'Mean LE':<10} {'Median LE':<10} {'Count':<6}")
print(f"  {'-'*34}")
for grade in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == grade]
    if len(subset) > 0:
        mean_le = subset["life_expectancy"].mean()
        median_le = subset["life_expectancy"].median()
        print(f"  {grade:<8} {mean_le:<10.1f} {median_le:<10.1f} {len(subset):<6}")

# ANOVA: group differences
groups = [
    analysis[analysis["dominant_grade"] == g]["life_expectancy"].values
    for g in ["A", "B", "C", "D"]
    if len(analysis[analysis["dominant_grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (life expectancy across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    → Significant difference between groups (p < 0.05)")

print("=" * 65)


# ─── 7. Visualization ───────────────────────────────────────────────────────
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle(
    "HOLC Redlining (1930s) and Current Life Expectancy in Chicago",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# --- Panel 1: Scatter plot – weighted grade score vs life expectancy ---
ax1 = axes[0, 0]
colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax1.scatter(
    analysis["weighted_grade_score"],
    analysis["life_expectancy"],
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)

# Add regression line
slope, intercept, r, p, se = stats.linregress(
    analysis["weighted_grade_score"], analysis["life_expectancy"]
)
x_line = np.linspace(1, 4, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel("Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11)
ax1.set_ylabel("Life Expectancy (years, 2010)", fontsize=11)
ax1.set_title(f"Weighted HOLC Score vs Life Expectancy\nr={r:.3f}, p={p:.4f}", fontsize=13)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax1.grid(True, alpha=0.3)

# --- Panel 2: Box plot – life expectancy by dominant grade ---
ax2 = axes[0, 1]
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == g]["life_expectancy"]
    if len(subset) > 0:
        grade_data.append(subset.values)
        grade_labels.append(f"Grade {g}")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Life Expectancy (years, 2010)", fontsize=11)
ax2.set_title("Life Expectancy by Dominant HOLC Grade", fontsize=13)
ax2.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Scatter – redlined fraction vs life expectancy ---
ax3 = axes[1, 0]
ax3.scatter(
    analysis["redlined_fraction"] * 100,
    analysis["life_expectancy"],
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
# Regression line
slope2, intercept2, r2, p2, se2 = stats.linregress(
    analysis["redlined_fraction"] * 100, analysis["life_expectancy"]
)
x_line2 = np.linspace(0, 100, 100)
ax3.plot(x_line2, slope2 * x_line2 + intercept2, "k--", linewidth=2, alpha=0.7)
ax3.set_xlabel("% of Community Area Redlined (Grade D)", fontsize=11)
ax3.set_ylabel("Life Expectancy (years, 2010)", fontsize=11)
ax3.set_title(
    f"Redlined Fraction vs Life Expectancy\nr={r2:.3f}, p={p2:.4f}", fontsize=13
)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Choropleth map – life expectancy with HOLC overlay ---
ax4 = axes[1, 1]
analysis_geo = analysis.copy()
analysis_geo = analysis_geo.to_crs(epsg=3857)
analysis_geo.plot(
    column="life_expectancy",
    cmap="RdYlGn",
    legend=True,
    ax=ax4,
    edgecolor="gray",
    linewidth=0.5,
    alpha=0.8,
    legend_kwds={"label": "Life Expectancy (years)", "shrink": 0.7},
)

# Overlay HOLC boundaries
holc_plot = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_plot[holc_plot["grade"] == grade]
    subset.boundary.plot(ax=ax4, color=COLOR_MAP[grade], linewidth=0.4, alpha=0.5)

ax4.set_axis_off()
ax4.set_title("Life Expectancy Map with HOLC Zone Boundaries", fontsize=13)

# Add legend for HOLC grades
legend_elements = [
    Patch(facecolor=c, edgecolor="black", label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(script_dir, "holc_life_expectancy_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  Saved figure to {output_path}")
plt.show()

print("\nDone!")
