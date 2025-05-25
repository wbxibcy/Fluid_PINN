import numpy as np
import matplotlib.pyplot as plt
import json
import os
from shapely.geometry import shape, Point, Polygon

def load_geojson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data

def extract_polygon_data(geojson_data):
    boundary = None
    holes = []

    for feature in geojson_data["features"]:
        geom = shape(feature["geometry"])
        ftype = feature["properties"].get("type")
        if ftype == "boundary":
            boundary = geom
        elif ftype == "obstacle":
            holes.append(geom)

    if boundary is None:
        raise ValueError("GeoJSON must contain a 'boundary' polygon")

    return boundary, holes

def create_structured_grid(boundary: Polygon, holes, nx, ny):
    minx, miny, maxx, maxy = boundary.bounds

    # 均匀结构网格
    x_base = np.linspace(minx, maxx, nx)
    y_base = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x_base, y_base)

    # 掩膜只保留在区域内的点
    mask = np.zeros(X.shape, dtype=bool)
    for j in range(Y.shape[0]):
        for i in range(X.shape[1]):
            pt = Point(X[j, i], Y[j, i])
            if boundary.contains(pt) and all(not hole.contains(pt) for hole in holes):
                mask[j, i] = True

    return X, Y, mask

def visualize_polygon_raw(boundary, holes):
    fig, ax = plt.subplots(figsize=(8, 8))
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', label='Boundary')
    for i, hole in enumerate(holes):
        hx, hy = hole.exterior.xy
        ax.plot(hx, hy, 'r--', label='Obstacle' if i == 0 else "")
    ax.set_aspect('equal')
    ax.set_title("Original Room Polygon (Before Meshing)")
    ax.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def visualize_structured_grid(X, Y, mask):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(X[mask], Y[mask], 'b.', markersize=1)
    ax.set_aspect('equal')
    ax.set_title("Uniform Structured Grid (Masked Inside Domain)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def save_npz_mesh(X, Y, mask, out_path="meshes/structured_grid.npz"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, mask=mask)
    print(f"✅ Structured grid saved: {out_path}")

if __name__ == "__main__":
    geojson_file = "modeling/room.geojson"
    geojson_data = load_geojson(geojson_file)

    boundary, holes = extract_polygon_data(geojson_data)

    visualize_polygon_raw(boundary, holes)

    X, Y, mask = create_structured_grid(boundary, holes, nx=500, ny=500)

    save_npz_mesh(X, Y, mask, out_path="meshes/structured_grid.npz")
    visualize_structured_grid(X, Y, mask)
