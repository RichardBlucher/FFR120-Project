import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.features import rasterize

# Load downloaded file from: https://geojson.io/#new&map=7.21/35.421/139.044/0/2
geojson_path = r'C:\Users\Olle\Desktop\FFR120\Project\map3.geojson' #geojson path
tokyo_boundary = gpd.read_file(geojson_path)

# Check for any invalid geometries
print("Are geometries valid?", tokyo_boundary.is_valid.all())

# Plot the boundary after reprojection
fig, ax = plt.subplots(figsize=(10, 10))
tokyo_boundary.plot(ax=ax, color='lightblue', edgecolor='black')
plt.title("Tokyo Boundary")
plt.show()

bounds = tokyo_boundary.total_bounds  # (minx, miny, maxx, maxy)
print(f"Bounds: {bounds}")  # print bounds

resolution = 0.001  # Size of each grid cell in meters ?

# Ensure resolution is appropriate
cols = int((bounds[2] - bounds[0]) / resolution)
rows = int((bounds[3] - bounds[1]) / resolution)

transform = rasterio.transform.from_bounds(*bounds, cols, rows)

# Rasterize the polygons with correct handling of the interior
geometry = [(geom, 1) for geom in tokyo_boundary.geometry]
raster = rasterize(
        shapes=geometry,
        out_shape=(rows, cols),
        transform=transform,
        fill=0,  # Default value for cells outside the geometry
        dtype=np.uint8,
        all_touched=True  # Ensure all touched pixels are filled
)

cmap = plt.cm.colors.ListedColormap(['blue', 'green'])

plt.imshow(raster, cmap=cmap)
plt.title("Rasterized Map")
plt.show()

print(raster)
print("Bounds:", bounds)
print("Raster matrix size:", raster.shape)

# Set specific cells in the raster matrix to a new value
raster[100:140, 560:600] = 2  # Mark a 40x40 block with the value 2

# Define colors for the values (0: blue, 1: green, 2: red)
cmap = mcolors.ListedColormap(['blue', 'green', 'red'])
bounds = [0, 1, 2, 3]  # Define boundaries for color mapping
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.imshow(raster, cmap=cmap, norm=norm)
plt.title("Raster Map with Custom Colors")
plt.show()

tmap = np.array(raster)
print(tmap.shape)
print(np.size(tmap, 0))
print(tmap)
