import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import contextily as cx

# Define the color mapping for HOLC grades
color_map = {
    'A': '#76a865',  # Green
    'B': '#7cbbe3',  # Blue
    'C': '#ffff00',  # Yellow
    'D': '#d9534f'   # Red
}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the GeoJSON file
file_path = os.path.join(script_dir, 'geojson.json')

# Read the GeoJSON file
try:
    gdf = gpd.read_file(file_path)
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
ax.set_title('HOLC Redlining Grades in Chicago', fontdict={'fontsize': '20', 'fontweight': '3'})

# Create custom legend handles
legend_elements = [Patch(facecolor=color, edgecolor='black', label=f'Grade {grade}') for grade, color in color_map.items()]

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
