import random

from obia.handlers import open_geotiff
from obia.segment import segment
import time

# for i in range(1, 212):
#     if i > 1:
#         break
all_numbers = list(range(1, 212))
completed = [210, 209, 208, 199, 196, 195, 186, 183, 181, 180, 179, 178, 177, 176, 175, 167, 166,
          163, 162, 161, 160, 147, 146, 69, 68, 55, 54, 53, 52, 40, 39, 28, 27, 18]
remaining = [num for num in all_numbers if num not in completed]

random_20 = random.sample(remaining, 20)
for i in random_20:
    start_time = time.time()
    raster_path = f"/mnt/c/tmp/output/output_{i}.tif"
    image = open_geotiff(raster_path)
    segmented_image = segment(
        image, segmentation_bands=[7,4,1],
        method="slic", n_segments=25000, compactness=10, max_num_iter=100, sigma=0.5, convert2lab=True, slic_zero=True
    )
    segmented_image.write_segments(f'/mnt/c/tmp/output/output_{i}.geojson')
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
