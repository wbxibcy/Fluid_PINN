import json
import matplotlib.pyplot as plt
from shapely.geometry import shape

with open("room.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

fig, ax = plt.subplots(figsize=(6, 6))

for feature in geojson_data["features"]:
    geom = shape(feature["geometry"])
    prop_type = feature["properties"]["type"]
    
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        if prop_type == "boundary":
            ax.plot(x, y, 'black', label='Room Boundary')
        elif prop_type == "obstacle":
            ax.fill(x, y, color='gray', alpha=0.5, label='Obstacle (Table)')
    
    elif geom.geom_type == "LineString":
        x, y = geom.xy
        if prop_type == "inlet":
            ax.plot(x, y, 'g-', linewidth=3, label='Inlet')
            ax.text(x[0] + 0.2, y[0], "Inlet", color='green')
        elif prop_type == "outlet":
            ax.plot(x, y, 'r-', linewidth=3, label='Outlet')
            ax.text(x[0] + 0.2, y[0], "Outlet", color='red')

ax.set_title("Room Layout with Inlet and Outlet")
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_aspect('equal')
ax.grid(True)

# 避免重复标签
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
