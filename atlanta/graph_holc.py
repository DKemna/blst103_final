"""
HOLC Redlining Map – Atlanta, Georgia
======================================
This script downloads and visualizes the 1930s HOLC redlining map for Atlanta.

Data source:
- HOLC redlining zones: Mapping Inequality (University of Richmond)
  https://dsl.richmond.edu/panorama/redlining/
"""

import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import contextily as cx
import requests
import os

# Define the color mapping for HOLC grades
color_map = {
    'A': '#76a865',  # Green  – "Best"
    'B': '#7cbbe3',  # Blue   – "Still Desirable"
    'C': '#ffff00',  # Yellow – "Definitely Declining"
    'D': '#d9534f'   # Red    – "Hazardous" (redlined)
}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Download / load HOLC data ──────────────────────────────────────────────
geojson_path = os.path.join(script_dir, 'geojson.json')

if not os.path.exists(geojson_path):
    print("Downloading Atlanta HOLC redlining data from Mapping Inequality...")
    holc_urls = [
        "https://dsl.richmond.edu/panorama/redlining/static/citiesData/GAAtlanta1938/geojson.json",
        "https://dsl.richmond.edu/panorama/redlining/static/downloads/geojson/GAAtlanta1938.geojson",
    ]
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    downloaded = False
    for url in holc_urls:
        try:
            resp = requests.get(url, timeout=60, headers=headers)
            if resp.status_code == 200 and len(resp.content) > 1000 and resp.text.lstrip().startswith('{'):
                with open(geojson_path, 'wb') as f:
                    f.write(resp.content)
                print(f"  Downloaded from {url}")
                downloaded = True
                break
        except Exception as e:
            print(f"  Could not download from {url}: {e}")
    if not downloaded:
        print("ERROR: Could not download Atlanta HOLC data.")
        print("  Please download manually from https://dsl.richmond.edu/panorama/redlining/")
        print(f"  Save the GeoJSON as: {geojson_path}")
        exit(1)
else:
    print(f"Using cached HOLC data: {geojson_path}")

# Read the GeoJSON file
try:
    gdf = gpd.read_file(geojson_path)
except Exception as e:
    print(f"Error reading the GeoJSON file: {e}")
    exit()

# Reproject to Web Mercator for contextily
gdf = gdf.to_crs(epsg=3857)

# Create a new figure and axes for the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Plot each grade with its corresponding color
for grade, color in color_map.items():
    gdf[gdf['grade'] == grade].plot(ax=ax, color=color, label=f'Grade {grade}', alpha=0.7)

# Add the basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

# Turn off the axis
ax.set_axis_off()

# Set plot title
ax.set_title('HOLC Redlining Grades in Atlanta', fontdict={'fontsize': '20', 'fontweight': '3'})

# Create custom legend handles
legend_elements = [
    Patch(facecolor=color, edgecolor='black', label=f'Grade {grade}')
    for grade, color in color_map.items()
]

# Add a legend with custom handles
ax.legend(handles=legend_elements, loc='upper left')

# Construct the full path for the output image
output_path = os.path.join(script_dir, 'holc_map.png')

# Save the plot to a file
try:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_path}")
except Exception as e:
    print(f"Error saving the map: {e}")

plt.show()
