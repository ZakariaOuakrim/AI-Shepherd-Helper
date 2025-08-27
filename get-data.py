import ee
import geemap

ee.Authenticate()
ee.Initialize()

# Define region
region = ee.Geometry.Rectangle([-5.2, 33.3, -4.8, 33.7])

# Load Sentinel-2
collection = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterBounds(region) \
    .filterDate('2023-06-01', '2023-06-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

image = collection.median()
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Download to local GeoTIFF
geemap.ee_export_image(ndvi, filename="ndvi_atlas.tif", scale=10, region=region)
