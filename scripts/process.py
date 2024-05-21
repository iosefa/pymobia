import geopandas as gpd
import time

from obia.handlers import open_geotiff
from obia.segment import segment
from obia.classify import classify

def reassign_class(x):
    if x not in keep_classes:
        return 0
    else:
        return x

start_time = time.time()
max_obs = 2000

training_path = "/mnt/c/tmp/output/Merge2.geojson"

keep_classes = [1, 2, 4, 8, 9, 12, 13, 16]
training_segments = gpd.read_file(training_path)
training_segments = training_segments.dropna()

training_segments = training_segments.drop(columns=['OBJECTID', 'Shape_Length', 'Shape_Area'])
training_segments['feature_class'] = training_segments['feature_class'].apply(reassign_class)

classes = training_segments['feature_class'].unique()

for cls in classes:
    count = training_segments[training_segments['feature_class'] == cls].shape[0]

    if count > max_obs:
        drop_indices = training_segments[training_segments['feature_class'] == cls].index[max_obs:]
        training_segments = training_segments.drop(drop_indices)



raster_path = "/mnt/c/tmp/output/output_163.tif"

image = open_geotiff(raster_path)
segmented_image = segment(
    image, segmentation_bands=[7,4,1],
    method="slic", n_segments=25000, compactness=10, max_num_iter=100, sigma=0.5, convert2lab=True, slic_zero=True
)
classified = classify(image, segmented_image, training_segments, method='mlp', compute_shap=True, solver='lbfgs')

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time}")
