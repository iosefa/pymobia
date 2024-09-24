import math
import os

import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.features import rasterize
from shapely import Polygon
from shapely.geometry import box
import geopandas as gpd

from obia.handlers.geotif import open_binary_geotiff_as_mask, open_geotiff
from obia.segmentation.segment_boundaries import create_segments


def join_segments(input_dir, output_filepath):
    all_segments = gpd.GeoDataFrame()

    for filename in os.listdir(input_dir):
        if filename.endswith("_tile.gpkg"):
            filepath = os.path.join(input_dir, filename)
            segments = gpd.read_file(filepath)
            all_segments = pd.concat([all_segments, segments], ignore_index=True)

    all_segments.to_file(output_filepath, driver="GPKG")


def get_raster_bbox(dataset):
    transform = dataset.GetGeoTransform()

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    min_x = transform[0]
    max_y = transform[3]
    max_x = min_x + width * transform[1]
    min_y = max_y + height * transform[5]

    return (min_x, min_y, max_x, max_y)


def create_tiled_segments(input_raster, input_mask, output_dir, tile_size=200, buffer=30):
    buffer = buffer * 2
    dataset = gdal.Open(input_raster)
    # todo: mask should be optional
    mask_dataset = gdal.Open(input_mask)

    # todo: dont mask segments that intersect with outer bbox
    outer_bbox = get_raster_bbox(mask_dataset)

    if not dataset or not mask_dataset:
        raise ValueError(f"Unable to open {input_raster} or {input_mask}")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    os.makedirs(output_dir, exist_ok=True)

    tile_index = 0

    all_black_segments = gpd.GeoDataFrame(columns=["geometry"])
    all_white_segments = gpd.GeoDataFrame(columns=["geometry"])

    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = (i // tile_size + j // tile_size) % 2 != 0

            if is_white_tile:
                i_offset = max(0, i - buffer)
                j_offset = max(0, j - buffer)
                if i == 0 or i == max(range(0, width, tile_size)):
                    w = min(tile_size + buffer, width - i_offset)
                else:
                    w = min(tile_size + buffer * 2, width - i_offset + buffer)

                if j == 0 or j == max(range(0, height, tile_size)):
                    h = min(tile_size + buffer, height - j_offset)
                else:
                    h = min(tile_size + buffer * 2, height - j_offset + buffer)
            else:
                i_offset = i
                j_offset = j
                w = min(tile_size, width - i_offset)
                h = min(tile_size, height - j_offset)

            # Create a memory dataset for the tile
            tile_transform = dataset.GetGeoTransform()
            tile_transform = (
                tile_transform[0] + i_offset * tile_transform[1],  # top left x
                tile_transform[1],  # w-e pixel resolution
                0,  # rotation, 0 if image is "north up"
                tile_transform[3] + j_offset * tile_transform[5],  # top left y
                0,  # rotation, 0 if image is "north up"
                tile_transform[5],  # n-s pixel resolution
            )

            label = "white" if is_white_tile else "black"
            image_output_filename = os.path.join(output_dir, f"{label}_tile_{j}_{i}.tif")
            mask_output_filename = os.path.join(output_dir, f"{label}_tile_{j}_{i}_mask.tif")

            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                image_output_filename,
                w, h,
                dataset.RasterCount,
                dataset.GetRasterBand(1).DataType,
                options=["COMPRESS=NONE"]
            )
            dst_ds.SetGeoTransform(tile_transform)
            dst_ds.SetProjection(dataset.GetProjection())

            mask_dst_ds = driver.Create(
                mask_output_filename,
                w, h,
                mask_dataset.RasterCount,
                mask_dataset.GetRasterBand(1).DataType,
                options=["COMPRESS=NONE"]
            )
            mask_dst_ds.SetGeoTransform(tile_transform)
            mask_dst_ds.SetProjection(mask_dataset.GetProjection())

            for band in range(1, dataset.RasterCount + 1):
                data = dataset.GetRasterBand(band).ReadAsArray(i_offset, j_offset, w, h)
                if data is not None:
                    dst_ds.GetRasterBand(band).WriteArray(data)

            for band in range(1, mask_dataset.RasterCount + 1):
                mask_data = mask_dataset.GetRasterBand(band).ReadAsArray(i_offset, j_offset, w, h)
                if mask_data is not None:
                    mask_dst_ds.GetRasterBand(band).WriteArray(mask_data)

            dst_ds = None
            mask_dst_ds = None


            if not is_white_tile:
                image = open_geotiff(image_output_filename)
                mask, _, _, _ = open_binary_geotiff_as_mask(mask_output_filename)

                pixel_area = 0.5 ** 2
                crown_area = math.pi * (6 ** 2)
                tree_area = mask.sum() * pixel_area
                n_crowns = round(tree_area / crown_area)
                print(n_crowns)

                try:
                    # todo pass as appropriate parameters and generalize
                    segmented_image = create_segments(
                        image=image,
                        mask=mask,
                        method="slic",
                        compactness=0.25,
                        sigma=0,
                        n_segments=n_crowns,
                        convert2lab=False,
                        slic_zero=True
                    )

                    bbox = segmented_image.total_bounds  # (minx, miny, maxx, maxy)
                    tile_boundary = box(bbox[0], bbox[1], bbox[2], bbox[3])

                    segmented_image_filtered = segmented_image[~segmented_image.intersects(tile_boundary.boundary)]
                    all_black_segments = pd.concat([all_black_segments, segmented_image_filtered], ignore_index=True)
                except ValueError:
                    print(f"empty tile: {image_output_filename}")

            tile_index += 1
        tile_index += 1

    tile_index = 0
    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = (i // tile_size + j // tile_size) % 2 != 0

            if is_white_tile:
                image_output_filename = os.path.join(output_dir, f"white_tile_{j}_{i}.tif")
                mask_output_filename = os.path.join(output_dir, f"white_tile_{j}_{i}_mask.tif")

                image = open_geotiff(image_output_filename)
                mask, mask_bbox, mask_transform, profile = open_binary_geotiff_as_mask(mask_output_filename)

                tile_polygon = box(*mask_bbox)

                corner_length = buffer / 2
                minx, miny, maxx, maxy = tile_polygon.bounds
                bottom_left_square = Polygon([
                    (minx, miny),
                    (minx + corner_length, miny),
                    (minx + corner_length, miny + corner_length),
                    (minx, miny + corner_length)
                ])
                bottom_right_square = Polygon([
                    (maxx - corner_length, miny),
                    (maxx, miny),
                    (maxx, miny + corner_length),
                    (maxx - corner_length, miny + corner_length)
                ])
                tile_polygon = tile_polygon.difference(bottom_left_square).difference(bottom_right_square)

                #todo correct
                intersecting_black_segments = all_black_segments[
                    all_black_segments.within(tile_polygon) | all_black_segments.overlaps(tile_polygon)
                    ]

                intersecting_white_segments = all_white_segments[
                    all_white_segments.within(tile_polygon) | all_white_segments.overlaps(tile_polygon)
                    ]

                if not intersecting_black_segments.empty or not intersecting_white_segments.empty:
                    overlapping_black_segments_for_mask = intersecting_black_segments[
                        intersecting_black_segments.overlaps(tile_polygon)
                    ]
                    overlapping_white_segments_for_mask = intersecting_white_segments[
                        intersecting_white_segments.overlaps(tile_polygon)
                    ]

                    overlapping_black_segments_for_delete = intersecting_black_segments[
                        intersecting_black_segments.within(tile_polygon)
                    ]
                    overlapping_white_segments_for_delete = intersecting_white_segments[
                        intersecting_white_segments.within(tile_polygon)
                    ]

                    indices_to_delete_black = overlapping_black_segments_for_delete.index
                    all_black_segments = all_black_segments.drop(indices_to_delete_black)

                    indices_to_delete_white = overlapping_white_segments_for_delete.index
                    all_white_segments = all_white_segments.drop(indices_to_delete_white)

                    # Step 1: Prepare geometries for white, black, and corners
                    combined_geometries = []

                    if not overlapping_white_segments_for_mask.empty:
                        white_geometries = [
                            (segment.geometry, 1) for _, segment in overlapping_white_segments_for_mask.iterrows()
                        ]
                        combined_geometries.extend(white_geometries)

                    # Add black segments geometries
                    if not overlapping_black_segments_for_mask.empty:
                        black_geometries = [
                            (segment.geometry, 1) for _, segment in overlapping_black_segments_for_mask.iterrows()
                        ]
                        combined_geometries.extend(black_geometries)

                    # Add corner geometries
                    corner_geometries = [(bottom_left_square, 1), (bottom_right_square, 1)]
                    combined_geometries.extend(corner_geometries)

                    mask_rasterized = rasterize(
                        combined_geometries,
                        out_shape=mask.shape,
                        transform=mask_transform,
                        fill=0,
                        default_value=1,
                        dtype=rasterio.uint8
                    )

                    # Update mask: set covered areas (where mask_rasterized == 1) to 0
                    mask[mask_rasterized == 1] = 0

                    # Save the updated mask
                    segmask_output_filename = mask_output_filename.replace(".tif", "_segmask.tif")
                    with rasterio.open(segmask_output_filename, 'w', **profile) as dst:
                        dst.write(mask.astype(rasterio.uint8), 1)

                    geometry_list = [geom[0] for geom in combined_geometries]  # Extract geometry part

                    # Create a GeoDataFrame with the geometries
                    combined_gdf = gpd.GeoDataFrame(geometry=geometry_list, crs=all_black_segments.crs)

                    # Save the GeoDataFrame to a GeoPackage
                    segmask_gpkg_filename = mask_output_filename.replace(".tif", "_segmask.gpkg")
                    combined_gdf.to_file(segmask_gpkg_filename, driver="GPKG")

                else:
                    print(f"No overlapping black segments found for tile ({i}, {j}).")

                pixel_area = 0.5 ** 2
                crown_area = math.pi * (6 ** 2)
                tree_area = mask.sum() * pixel_area
                n_crowns = round(tree_area / crown_area)
                print(n_crowns)

                # Perform segmentation for the white tile
                # todo pass as appropriate parameters and generalize
                try:
                    segmented_image = create_segments(
                        image=image,
                        mask=mask,
                        method="slic",
                        compactness=0.25,
                        sigma=0,
                        n_segments=n_crowns,
                        convert2lab=False,
                        slic_zero=True
                    )
                    all_white_segments = pd.concat([all_white_segments, segmented_image], ignore_index=True)
                except ValueError:
                    print(f"empty tile: {image_output_filename}")

            tile_index += 1
        tile_index += 1

    all_segments = pd.concat([all_black_segments, all_white_segments], ignore_index=True)
    all_segments.to_file(os.path.join(output_dir, "segments.gpkg"), driver="GPKG")


def create_tiled_segments_naive(input_raster, input_mask, output_dir, tile_size=200):
    # Open the input raster and mask using GDAL
    dataset = gdal.Open(input_raster)
    # todo: mask should be optional
    mask_dataset = gdal.Open(input_mask)

    # todo: dont mask segments that intersect with outer bbox
    outer_bbox = get_raster_bbox(mask_dataset)

    if not dataset or not mask_dataset:
        raise ValueError(f"Unable to open {input_raster} or {input_mask}")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    os.makedirs(output_dir, exist_ok=True)

    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            i_offset = i
            j_offset = j
            w = min(tile_size, width - i_offset)
            h = min(tile_size, height - j_offset)

            # Create a memory dataset for the tile
            tile_transform = dataset.GetGeoTransform()
            tile_transform = (
                tile_transform[0] + i_offset * tile_transform[1],  # top left x
                tile_transform[1],  # w-e pixel resolution
                0,  # rotation, 0 if image is "north up"
                tile_transform[3] + j_offset * tile_transform[5],  # top left y
                0,  # rotation, 0 if image is "north up"
                tile_transform[5],  # n-s pixel resolution
            )

            image_output_filename = os.path.join(output_dir, f"tile_{j}_{i}.tif")
            mask_output_filename = os.path.join(output_dir, f"mask_{j}_{i}.tif")

            # Create a new GeoTIFF file for the tile (image and mask)
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                image_output_filename,
                w, h,
                dataset.RasterCount,
                dataset.GetRasterBand(1).DataType,
                options=["COMPRESS=NONE"]  # Disable compression
            )
            dst_ds.SetGeoTransform(tile_transform)
            dst_ds.SetProjection(dataset.GetProjection())

            mask_dst_ds = driver.Create(
                mask_output_filename,
                w, h,
                mask_dataset.RasterCount,
                mask_dataset.GetRasterBand(1).DataType,
                options=["COMPRESS=NONE"]  # Disable compression
            )
            mask_dst_ds.SetGeoTransform(tile_transform)
            mask_dst_ds.SetProjection(mask_dataset.GetProjection())

            # Write the tile data to the file (for both image and mask)
            for band in range(1, dataset.RasterCount + 1):
                data = dataset.GetRasterBand(band).ReadAsArray(i_offset, j_offset, w, h)
                if data is not None:
                    dst_ds.GetRasterBand(band).WriteArray(data)

            for band in range(1, mask_dataset.RasterCount + 1):
                mask_data = mask_dataset.GetRasterBand(band).ReadAsArray(i_offset, j_offset, w, h)
                if mask_data is not None:
                    mask_dst_ds.GetRasterBand(band).WriteArray(mask_data)

            # Close the datasets to flush data to disk
            dst_ds = None
            mask_dst_ds = None

            image = open_geotiff(image_output_filename)
            mask, _, _, _ = open_binary_geotiff_as_mask(mask_output_filename)

            pixel_area = 0.5 ** 2
            crown_area = math.pi * (6 ** 2)
            tree_area = mask.sum() * pixel_area
            n_crowns = round(tree_area / crown_area)
            print(n_crowns)

            # todo pass as appropriate parameters and generalize
            try:
                segmented_image = create_segments(
                    image=image,
                    mask=mask,
                    method="slic",
                    compactness=0.25,
                    sigma=0,
                    n_segments=n_crowns,
                    convert2lab=False,
                    slic_zero=True
                )

                segmented_output_filename = os.path.join(output_dir, f"segments_{j}_{i}_tile.gpkg")
                segmented_image.to_file(segmented_output_filename, driver="GPKG")
            except ValueError:
                print(f"empty tile: {image_output_filename}")

    join_segments(output_dir, os.path.join(output_dir, "segments.gpkg"))
