"""
HOLC Redlining and K-8 Education Correlation Analysis
=====================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Chicago and current K-8 school performance metrics (test scores).

Data sources:
- HOLC redlining zones: local geojson.json
- CPS School Profile Information: Chicago Data Portal (multiple years)
- Community area boundaries: Chicago Data Portal (igwz-8jzy)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from shapely.geometry import Point
import requests
import os
import re
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

# CPS datasets to try — ordered so test-score-rich sets come first
CPS_DATASETS = [
    ("9xs2-f89t", "SY1617"),  # School Profile 2016-17  (has NWEA metrics)
    ("2m8w-izji", "SY1819"),  # School Profile 2018-19
    ("kh4r-387c", "SY2324"),  # School Profile 2023-24  (has overall_rating)
]

# Mapping for CPS overall_rating → numeric score (higher = better)
RATING_MAP = {
    "Level 1+": 90, "Level 1": 80, "Level 2+": 65,
    "Level 2": 50, "Level 3": 30,
    "INTENSIVE": 20, "PROBATION": 10,
}


def fetch_cps_data():
    """Try multiple CPS dataset endpoints and return the first that works."""
    for dataset_id, year in CPS_DATASETS:
        try:
            url = (
                f"https://data.cityofchicago.org/resource/{dataset_id}.json"
                f"?$limit=2000"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) > 10:
                    print(f"  Loaded {len(data)} schools from dataset {year} ({dataset_id})")
                    return pd.DataFrame(data), year
        except Exception as e:
            print(f"  Warning: could not fetch {dataset_id} ({year}): {e}")
    return None, None


def find_column(df, patterns, label=""):
    """Find the first column matching any of the given regex patterns."""
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, re.IGNORECASE):
                return c
    return None


def find_score_columns(df):
    """
    Auto-detect numeric test score columns from CPS school profile data.

    Returns:
        primary_cols: list of columns for reading + math attainment
        all_score_cols: all detected score-related columns
    """
    score_patterns = [
        # Grade-level attainment (CPS progress reports)
        r"gr3_5_grade_level",
        r"gr6_8_grade_level",
        r"pk_2_literacy",
        r"pk_2_math",
        # ISAT exceeding percentages  (0-100 scale)
        r"isat_exceeding",
        # NWEA MAP test scores
        r"nwea.*read.*pct",
        r"nwea.*math.*pct",
        r"nwea.*read.*attain",
        r"nwea.*math.*attain",
        # Attainment / proficiency
        r"attain.*read.*pct",
        r"attain.*math.*pct",
        r"attainment.*reading",
        r"attainment.*math",
        # PARCC / IAR (newer Illinois tests)
        r"parcc.*ela",
        r"parcc.*math",
        r"iar.*ela",
        r"iar.*math",
        # Generic patterns
        r"reading.*pct",
        r"math.*pct",
        r"proficien.*pct",
        r"ela.*pct",
        r"reading_score",
        r"math_score",
        r"adequate_yearly_progress",
    ]

    all_score_cols = []
    for pattern in score_patterns:
        for c in df.columns:
            if re.search(pattern, c, re.IGNORECASE) and c not in all_score_cols:
                # Verify column is numeric-like
                sample = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(sample) > 5:
                    all_score_cols.append(c)

    # Exclude growth / value-add columns from primary metrics (different scale)
    exclude_patterns = [r"value_add", r"growth", r"keep_pace", r"color"]
    primary_cols = [
        c for c in all_score_cols
        if not any(re.search(ep, c, re.IGNORECASE) for ep in exclude_patterns)
    ]

    if len(primary_cols) == 0:
        primary_cols = all_score_cols[:4]

    return primary_cols, all_score_cols


def filter_k8_schools(df):
    """Filter DataFrame to keep only K-8 (elementary + middle) schools."""
    cols = df.columns.tolist()

    # Strategy 0: Use elementary_or_high_school column (SY1617 dataset)
    eh_col = find_column(df, [r"elementary_or_high_school", r"elem.*or.*high"])
    if eh_col:
        before = len(df)
        vals = df[eh_col].astype(str).str.upper().str.strip()
        df = df[vals.isin(["ES", "ELEMENTARY", "E"])].copy()
        print(f"  Filtered K-8: {len(df)} elementary schools (from {before} via '{eh_col}')")
        return df

    # Strategy 1: Use is_high_school flag to exclude HS
    hs_col = find_column(df, [r"is_high_school", r"is_high", r"highschool"])
    if hs_col:
        before = len(df)
        df[hs_col] = df[hs_col].astype(str).str.lower().str.strip()
        df = df[~df[hs_col].isin(["true", "1", "yes", "y"])].copy()
        print(f"  Filtered K-8: {len(df)} schools (excluded {before - len(df)} high schools via '{hs_col}')")
        return df

    # Strategy 2: Use primary_category column (ES=Elementary, MS=Middle, HS=High)
    cat_col = find_column(df, [r"primary_category", r"school_type", r"governance"])
    if cat_col:
        before = len(df)
        cat_values = df[cat_col].astype(str).str.upper()
        k8_mask = cat_values.isin(["ES", "MS", "ELEMENTARY", "MIDDLE"])
        if k8_mask.sum() > 0:
            df = df[k8_mask].copy()
            print(f"  Filtered K-8: {len(df)} schools (from {before} via '{cat_col}')")
            return df

    # Strategy 3: Use grade range columns
    grade_hi_col = find_column(df, [r"grades_offered.*hi", r"grade.*end", r"highest_grade"])
    if grade_hi_col:
        before = len(df)
        grade_hi = pd.to_numeric(df[grade_hi_col], errors="coerce")
        df = df[grade_hi <= 8].copy()
        print(f"  Filtered K-8: {len(df)} schools (from {before} via '{grade_hi_col}')")
        return df

    print("  WARNING: Could not identify school type column — using all schools")
    return df


# ─── 1. Load HOLC redlining data ────────────────────────────────────────────
print("Loading HOLC redlining data...")
holc = gpd.read_file(os.path.join(script_dir, "geojson.json"))
holc = holc[holc["grade"].isin(["A", "B", "C", "D"])].copy()
holc = holc.to_crs(epsg=4326)

print(f"  Loaded {len(holc)} HOLC zones")
for g in ["A", "B", "C", "D"]:
    print(f"    Grade {g}: {(holc['grade'] == g).sum()} zones")


# ─── 2. Download CPS school performance data ────────────────────────────────
print("\nDownloading CPS school performance data...")
schools_df, dataset_year = fetch_cps_data()

if schools_df is None:
    print("ERROR: Could not download CPS school data from any source.")
    exit(1)


# ─── 3. Identify and prepare columns ────────────────────────────────────────
print("\nPreparing data columns...")
cols = schools_df.columns.tolist()

# --- Location columns ---
lat_col = find_column(schools_df, [r"school_lat", r"^latitude$", r"lat$"])
lon_col = find_column(schools_df, [r"school_lon", r"^longitude$", r"lon$", r"lng$"])

# Fallback: check for nested location/geometry fields
if lat_col is None or lon_col is None:
    for c in cols:
        if "location" in c.lower() or "the_geom" in c.lower():
            sample = schools_df[c].dropna()
            if len(sample) > 0:
                val = sample.iloc[0]
                if isinstance(val, dict):
                    if "latitude" in val:
                        schools_df["_lat"] = schools_df[c].apply(
                            lambda x: float(x["latitude"])
                            if isinstance(x, dict) and "latitude" in x
                            else np.nan
                        )
                        schools_df["_lon"] = schools_df[c].apply(
                            lambda x: float(x["longitude"])
                            if isinstance(x, dict) and "longitude" in x
                            else np.nan
                        )
                        lat_col, lon_col = "_lat", "_lon"
                        break
                    elif "coordinates" in val:
                        schools_df["_lon"] = schools_df[c].apply(
                            lambda x: float(x["coordinates"][0])
                            if isinstance(x, dict) and "coordinates" in x
                            else np.nan
                        )
                        schools_df["_lat"] = schools_df[c].apply(
                            lambda x: float(x["coordinates"][1])
                            if isinstance(x, dict) and "coordinates" in x
                            else np.nan
                        )
                        lat_col, lon_col = "_lat", "_lon"
                        break

if lat_col is None or lon_col is None:
    print(f"ERROR: Could not find latitude/longitude columns")
    print(f"  Available columns: {cols}")
    exit(1)

print(f"  Location columns: {lat_col}, {lon_col}")

# --- School name column ---
name_col = find_column(schools_df, [r"short_name", r"long_name", r"school_name", r"name"])
if name_col:
    print(f"  School name column: {name_col}")

# --- Filter to K-8 schools ---
schools_df = filter_k8_schools(schools_df)

# --- Test score columns ---
primary_score_cols, all_score_cols = find_score_columns(schools_df)

if len(primary_score_cols) == 0 and len(all_score_cols) == 0:
    # Fallback: convert overall_rating to a numeric performance score
    rating_col = find_column(schools_df, [r"overall_rating", r"rating_status"])
    if rating_col:
        raw = schools_df[rating_col].astype(str).str.strip()
        mapped = raw.map(RATING_MAP)
        # If the mapping worked for some rows, use it
        if mapped.notna().sum() > 5:
            schools_df["_overall_rating_num"] = mapped
            primary_score_cols = ["_overall_rating_num"]
            print(f"  Using '{rating_col}' mapped to numeric scale as performance metric")
            value_counts = raw.value_counts()
            for val, cnt in value_counts.items():
                num = RATING_MAP.get(val, "?")
                print(f"      {val} → {num}  ({cnt} schools)")
        else:
            # Try direct numeric conversion
            numeric_ratings = pd.to_numeric(schools_df[rating_col], errors="coerce")
            if numeric_ratings.notna().sum() > 5:
                schools_df["_overall_rating_num"] = numeric_ratings
                primary_score_cols = ["_overall_rating_num"]
                print(f"  Using numeric '{rating_col}' as performance metric")
    if len(primary_score_cols) == 0:
        print("ERROR: No test score or performance columns detected.")
        print(f"  Available columns: {cols}")
        exit(1)

print(f"  Primary test score columns ({len(primary_score_cols)}):")
for c in primary_score_cols:
    n_valid = pd.to_numeric(schools_df[c], errors="coerce").dropna().shape[0]
    print(f"    • {c} ({n_valid} valid values)")

if len(all_score_cols) > len(primary_score_cols):
    print(f"  Additional score columns detected: {[c for c in all_score_cols if c not in primary_score_cols]}")

# --- Convert columns to numeric ---
for c in primary_score_cols:
    schools_df[c] = pd.to_numeric(schools_df[c], errors="coerce")

schools_df[lat_col] = pd.to_numeric(schools_df[lat_col], errors="coerce")
schools_df[lon_col] = pd.to_numeric(schools_df[lon_col], errors="coerce")

# --- Composite score (mean of primary score columns) ---
schools_df["composite_score"] = schools_df[primary_score_cols].mean(axis=1)
schools_df = schools_df.dropna(subset=["composite_score", lat_col, lon_col])

print(f"\n  Schools with valid test scores and location: {len(schools_df)}")
print(
    f"  Composite score range: "
    f"{schools_df['composite_score'].min():.1f} – {schools_df['composite_score'].max():.1f}"
)


# ─── 4. Geocode schools and spatial join with HOLC zones ────────────────────
print("\nSpatial join: assigning HOLC grade to each school...")

geometry = [Point(xy) for xy in zip(schools_df[lon_col], schools_df[lat_col])]
schools_gdf = gpd.GeoDataFrame(schools_df, geometry=geometry, crs="EPSG:4326")

# Spatial join: find which HOLC zone each school falls within
schools_holc = gpd.sjoin(
    schools_gdf,
    holc[["grade", "geometry"]],
    how="inner",
    predicate="within",
)

# Handle schools matched to multiple overlapping zones — keep the one
# with the smallest (worst) grade to be conservative, or just drop dupes
schools_holc = schools_holc.drop_duplicates(subset=[lat_col, lon_col], keep="first")

print(f"  K-8 schools within HOLC zones: {len(schools_holc)}")

if len(schools_holc) < 5:
    print("ERROR: Too few schools matched to HOLC zones.")
    print("  This may indicate a CRS or coordinate alignment issue.")
    exit(1)

schools_holc["grade_num"] = schools_holc["grade"].map(GRADE_NUM)

# Summary by grade
print(f"\n  Schools by HOLC grade:")
for g in ["A", "B", "C", "D"]:
    subset = schools_holc[schools_holc["grade"] == g]
    if len(subset) > 0:
        mean_s = subset["composite_score"].mean()
        print(f"    Grade {g}: {len(subset):>3} schools,  mean composite score = {mean_s:.1f}")


# ─── 5. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL RESULTS — HOLC Grade vs K-8 Test Scores")
print("=" * 70)

# --- Pearson correlation: numeric HOLC grade vs composite score ---
r_pearson, p_pearson = stats.pearsonr(
    schools_holc["grade_num"], schools_holc["composite_score"]
)
print(f"\n  Pearson correlation (HOLC grade ↔ composite test score):")
print(f"    r = {r_pearson:.4f},  p = {p_pearson:.6f}")
if p_pearson < 0.05:
    print(f"    ✓ Statistically significant (p < 0.05)")
if r_pearson < 0:
    print(f"    → Negative: worse HOLC grades associated with LOWER test scores")
else:
    print(f"    → Positive: worse HOLC grades associated with higher test scores")

# --- Spearman rank correlation ---
r_spear, p_spear = stats.spearmanr(
    schools_holc["grade_num"], schools_holc["composite_score"]
)
print(f"\n  Spearman rank correlation:")
print(f"    rₛ = {r_spear:.4f},  p = {p_spear:.6f}")

# --- Individual test score columns ---
if len(primary_score_cols) > 1:
    print(f"\n  Per-metric correlations:")
    for col in primary_score_cols:
        valid = schools_holc.dropna(subset=[col])
        if len(valid) > 5:
            r_i, p_i = stats.pearsonr(valid["grade_num"], valid[col])
            sig = " *" if p_i < 0.05 else ""
            print(f"    {col:<45s}  r={r_i:+.4f}  p={p_i:.4f}{sig}")

# --- Mean scores by HOLC grade ---
print(f"\n  Mean composite test score by HOLC grade:")
print(f"  {'Grade':<8} {'Mean':>8} {'Median':>8} {'Std Dev':>8} {'N':>5}")
print(f"  {'-' * 40}")
for grade in ["A", "B", "C", "D"]:
    subset = schools_holc[schools_holc["grade"] == grade]["composite_score"]
    if len(subset) > 0:
        print(
            f"  {grade:<8} {subset.mean():>8.1f} {subset.median():>8.1f} "
            f"{subset.std():>8.1f} {len(subset):>5}"
        )

# --- One-way ANOVA ---
groups = [
    schools_holc[schools_holc["grade"] == g]["composite_score"].values
    for g in ["A", "B", "C", "D"]
    if len(schools_holc[schools_holc["grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (composite score across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    ✓ Significant difference between groups (p < 0.05)")

# --- Point-biserial: redlined (D) vs not ---
schools_holc["is_redlined"] = (schools_holc["grade"] == "D").astype(int)
r_pb, p_pb = stats.pointbiserialr(
    schools_holc["is_redlined"], schools_holc["composite_score"]
)
print(f"\n  Point-biserial correlation (redlined D vs other grades):")
print(f"    r_pb = {r_pb:.4f},  p = {p_pb:.6f}")
if p_pb < 0.05:
    print(f"    ✓ Significant difference between redlined and non-redlined zones")

# --- Effect size: Cohen's d (Grade A vs Grade D) ---
a_scores = schools_holc[schools_holc["grade"] == "A"]["composite_score"]
d_scores = schools_holc[schools_holc["grade"] == "D"]["composite_score"]
if len(a_scores) > 1 and len(d_scores) > 1:
    pooled_std = np.sqrt(
        ((len(a_scores) - 1) * a_scores.std() ** 2 + (len(d_scores) - 1) * d_scores.std() ** 2)
        / (len(a_scores) + len(d_scores) - 2)
    )
    if pooled_std > 0:
        cohens_d = (a_scores.mean() - d_scores.mean()) / pooled_std
        print(f"\n  Cohen's d (Grade A vs Grade D):")
        print(f"    d = {cohens_d:.4f}")
        if abs(cohens_d) >= 0.8:
            print(f"    → Large effect size")
        elif abs(cohens_d) >= 0.5:
            print(f"    → Medium effect size")
        elif abs(cohens_d) >= 0.2:
            print(f"    → Small effect size")

print("=" * 70)


# ─── 6. Visualization ───────────────────────────────────────────────────────
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle(
    "HOLC Redlining (1930s) and K-8 School Performance in Chicago",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# --- Panel 1: Scatter – HOLC grade vs composite test score ---
ax1 = axes[0, 0]
np.random.seed(42)
jitter = np.random.normal(0, 0.08, len(schools_holc))
colors = [COLOR_MAP[g] for g in schools_holc["grade"]]
ax1.scatter(
    schools_holc["grade_num"] + jitter,
    schools_holc["composite_score"],
    c=colors,
    edgecolors="black",
    s=60,
    alpha=0.7,
    zorder=5,
    linewidths=0.5,
)
# Regression line
slope, intercept, r, p, se = stats.linregress(
    schools_holc["grade_num"], schools_holc["composite_score"]
)
x_line = np.linspace(0.5, 4.5, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel(
    "HOLC Grade (1 = A/Best → 4 = D/Hazardous)", fontsize=11
)
ax1.set_ylabel("Composite Test Score (%)", fontsize=11)
ax1.set_title(
    f"HOLC Grade vs K-8 Composite Test Score\nr = {r:.3f}, p = {p:.4f}",
    fontsize=13,
)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["A\n(Best)", "B\n(Desirable)", "C\n(Declining)", "D\n(Hazardous)"])
ax1.grid(True, alpha=0.3)

# --- Panel 2: Box plot – score distributions by HOLC grade ---
ax2 = axes[0, 1]
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = schools_holc[schools_holc["grade"] == g]["composite_score"]
    if len(subset) > 0:
        grade_data.append(subset.values)
        grade_labels.append(f"Grade {g}\n(n={len(subset)})")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Composite Test Score (%)", fontsize=11)
ax2.set_title("K-8 Test Score Distribution by HOLC Grade", fontsize=13)
ax2.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Bar chart – mean scores with standard error bars ---
ax3 = axes[1, 0]
means = []
sems = []
bar_labels = []
bar_colors = []
for g in ["A", "B", "C", "D"]:
    subset = schools_holc[schools_holc["grade"] == g]["composite_score"]
    if len(subset) > 0:
        means.append(subset.mean())
        sems.append(subset.std() / np.sqrt(len(subset)))
        bar_labels.append(f"Grade {g}")
        bar_colors.append(COLOR_MAP[g])

bars = ax3.bar(
    bar_labels,
    means,
    yerr=sems,
    color=bar_colors,
    edgecolor="black",
    alpha=0.8,
    capsize=8,
    linewidth=1,
)
for bar, m in zip(bars, means):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{m:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )
ax3.set_ylabel("Mean Composite Test Score (%)", fontsize=11)
ax3.set_title("Mean K-8 Test Scores by HOLC Grade (± SE)", fontsize=13)
ax3.grid(True, alpha=0.3, axis="y")

# --- Panel 4: Map – schools on HOLC redlining zones ---
ax4 = axes[1, 1]
holc_proj = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_proj[holc_proj["grade"] == grade]
    subset.plot(ax=ax4, color=COLOR_MAP[grade], alpha=0.3, edgecolor="gray", linewidth=0.3)

# Plot schools colored by composite test score
schools_proj = schools_holc.to_crs(epsg=3857)
sc = ax4.scatter(
    schools_proj.geometry.x,
    schools_proj.geometry.y,
    c=schools_proj["composite_score"],
    cmap="RdYlGn",
    edgecolors="black",
    s=30,
    alpha=0.9,
    zorder=5,
    linewidths=0.3,
)
cbar = plt.colorbar(sc, ax=ax4, shrink=0.7)
cbar.set_label("Composite Test Score (%)", fontsize=10)
ax4.set_axis_off()
ax4.set_title("K-8 Schools on HOLC Zones\n(dot color = test score)", fontsize=13)

# HOLC grade legend
legend_elements = [
    Patch(facecolor=c, edgecolor="black", alpha=0.5, label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(script_dir, "holc_k8_education_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  Saved figure to {output_path}")
plt.show()

print("\nDone!")
