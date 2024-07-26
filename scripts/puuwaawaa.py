import math
import os

import geopandas as gpd
import numpy as np

from obia.handlers import open_geotiff
from obia.segment import segment
from tqdm import tqdm
import glob

from osgeo import gdal


def open_binary_raster_as_mask(mask_path):
    raster = gdal.Open(mask_path, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    mask_array = band.ReadAsArray().astype(bool)
    return mask_array


tile_index_path = '/home/milo/repos/pd-hawaii-lidar/output/tileindex.gpkg'
tile_index = gpd.read_file(tile_index_path)

tile_directory_path = '/home/milo/repos/pd-hawaii-lidar/tiles/*.tif'
all_tiles = glob.glob(tile_directory_path)

mask_directory_path = '/home/milo/repos/pd-hawaii-lidar/masks/*.tif'
all_masks = glob.glob(mask_directory_path)

for tile_path, mask_path in tqdm(zip(all_tiles, all_masks)):
    tile = tile_index[tile_index['location'] == tile_path]

    intersections = gpd.overlay(tile, tile_index, how='intersection', keep_geom_type=False)

    if not intersections.empty:
        # assuming the filename without extension could be used
        filename = tile_path.split('/')[-1].split('.')[0]
        filepath = f'/home/milo/repos/obia/scripts/segments/seg_{filename}.gpkg'

        if os.path.exists(filepath):
            continue

        image = open_geotiff(tile_path)
        mask = open_binary_raster_as_mask(mask_path)
        print(tile_path, mask_path)

        if mask.sum() == 0:
            continue
        pixel_area = 0.5 ** 2
        crown_area = math.pi * (5 ** 2)
        tree_area = mask.sum() * pixel_area
        n_crowns = round(tree_area / crown_area)
        try:
            segmented_image = segment(
                image, segmentation_bands=[4, 5, 2], statistics_bands=[0, 1, 2, 3, 4, 5],
                calc_mean=True, calc_variance=True, calc_skewness=False, calc_kurtosis=False,
                calc_contrast=True, calc_dissimilarity=False, calc_homogeneity=False, calc_ASM=False, calc_energy=False,
                calc_correlation=True,
                method="slic", n_segments=n_crowns, compactness=0.01, max_num_iter=100, sigma=0, convert2lab=False,
                slic_zero=True, mask=mask)

            segmented_image.write_segments(f'/home/milo/repos/obia/scripts/segments/seg_{filename}.gpkg')
        except Exception as e:
            print(e)
            continue
