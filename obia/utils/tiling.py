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

    tile_index = 0

    all_black_segments = gpd.GeoDataFrame(columns=["geometry"])
    all_white_segments = gpd.GeoDataFrame(columns=["geometry"])

    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = tile_index % 2 != 0

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

            # Determine the label for the output filename
            label = "white" if is_white_tile else "black"
            image_output_filename = os.path.join(output_dir, f"{label}_tile_{j}_{i}.tif")
            mask_output_filename = os.path.join(output_dir, f"{label}_tile_{j}_{i}_mask.tif")

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


            # If it's a black tile, perform segmentation and save the result
            if not is_white_tile:
                # Read the image and mask data
                image = open_geotiff(image_output_filename)
                mask, _ = open_binary_geotiff_as_mask(mask_output_filename)

                pixel_area = 0.5 ** 2
                crown_area = math.pi * (6 ** 2)
                tree_area = mask.sum() * pixel_area
                n_crowns = round(tree_area / crown_area)
                print(n_crowns)

                # Perform segmentation
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

                    # Remove segments that intersect with the tile boundary
                    # Keep only segments that do not intersect with the boundary
                    segmented_image_filtered = segmented_image[~segmented_image.intersects(tile_boundary.boundary)]
                    print("filtered black segments shape", segmented_image_filtered.shape)
                    # Add the filtered segments to the all_black_segments GeoDataFrame
                    all_black_segments = pd.concat([all_black_segments, segmented_image_filtered], ignore_index=True)

                    # Save the filtered segmented image with appropriate name
                    segmented_output_filename = os.path.join(output_dir, f"segments_black_{j}_{i}_tile.gpkg")
                    segmented_image_filtered.to_file(segmented_output_filename, driver="GPKG")

                except ValueError:
                    print(f"empty tile: {image_output_filename}")

            tile_index += 1
        tile_index += 1

    # Second loop: segment the white tiles
    tile_index = 0
    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = tile_index % 2 != 0

            if is_white_tile:
                i_offset = max(0, i - buffer)
                j_offset = max(0, j - buffer)
                if i == 0 or i == width - tile_size:
                    w = min(tile_size + buffer, width - i_offset)
                else:
                    w = min(tile_size + buffer * 2, width - i_offset + buffer)

                if j == 0 or j == height - tile_size:
                    h = min(tile_size + buffer, height - j_offset)
                else:
                    h = min(tile_size + buffer * 2, height - j_offset + buffer)

                image_output_filename = os.path.join(output_dir, f"white_tile_{j}_{i}.tif")
                mask_output_filename = os.path.join(output_dir, f"white_tile_{j}_{i}_mask.tif")

                # Read the image and mask data for the white tile
                image = open_geotiff(image_output_filename)
                mask, mask_bbox = open_binary_geotiff_as_mask(mask_output_filename)

                tile_polygon = box(*mask_bbox)  # Create a Shapely polygon from the bbox

                # Update mask with overlapping black segments that intersect with the tile polygon
                overlapping_black_segments = all_black_segments[all_black_segments.intersects(tile_polygon)]
                overlapping_white_segments = all_white_segments[all_white_segments.intersects(tile_polygon)]

                print(f"Overlapping segments: {len(overlapping_black_segments)}, {len(overlapping_white_segments)}")

                if overlapping_black_segments.empty:
                    print("No overlapping segments found.")

                # Ensure there are valid geometries to rasterize
                if not overlapping_black_segments.empty:
                    valid_geometries = overlapping_black_segments.geometry.is_valid
                    overlapping_black_segments = overlapping_black_segments[valid_geometries]
                    print(f"Overlapping valid segments: {len(overlapping_black_segments)}")

                    segmask_gpkg_filename = mask_output_filename.replace(".tif", "_segmask.gpkg")
                    overlapping_black_segments.to_file(segmask_gpkg_filename, driver="GPKG")

                    if not overlapping_white_segments.empty:
                        # Convert the geometries to a list of (geometry, value) tuples for rasterization
                        with rasterio.open(mask_output_filename) as src:
                            mask_transform = src.transform

                        # Convert the geometries to a list of (geometry, value) tuples for rasterization
                        geometries = [(segment.geometry, 1) for _, segment in overlapping_white_segments.iterrows()]

                        # Rasterize the geometries onto the mask grid using the correct transform
                        mask_rasterized = rasterize(
                            geometries,
                            out_shape=mask.shape,
                            transform=mask_transform,  # Use the transform from the mask dataset
                            fill=0,
                            default_value=1,
                            dtype=rasterio.uint8
                        )

                        mask[mask_rasterized == 1] = 0

                    if not overlapping_black_segments.empty:
                        # Convert the geometries to a list of (geometry, value) tuples for rasterization
                        with rasterio.open(mask_output_filename) as src:
                            mask_transform = src.transform

                        # Convert the geometries to a list of (geometry, value) tuples for rasterization
                        geometries = [(segment.geometry, 1) for _, segment in overlapping_black_segments.iterrows()]

                        # Rasterize the geometries onto the mask grid using the correct transform
                        mask_rasterized = rasterize(
                            geometries,
                            out_shape=mask.shape,
                            transform=mask_transform,  # Use the transform from the mask dataset
                            fill=0,
                            default_value=1,
                            dtype=rasterio.uint8
                        )

                        mask[mask_rasterized == 1] = 0

                        # Get the bounds of the tile polygon
                        minx, miny, maxx, maxy = tile_polygon.bounds

                        # Calculate the size of the bottom corners
                        corner_width = buffer/2
                        corner_height = buffer/2

                        # Bottom-left corner polygon (using calculated corner size)
                        bottom_left_polygon = Polygon([
                            (minx, miny),
                            (minx + corner_width, miny),
                            (minx + corner_width, miny + corner_height),
                            (minx, miny + corner_height)
                        ])

                        # Bottom-right corner polygon (using calculated corner size)
                        bottom_right_polygon = Polygon([
                            (maxx - corner_width, miny),
                            (maxx, miny),
                            (maxx, miny + corner_height),
                            (maxx - corner_width, miny + corner_height)
                        ])

                        # Create a GeoDataFrame with both corners
                        if i == 0:
                            corners_gdf = gpd.GeoDataFrame({
                                'geometry': [bottom_right_polygon],
                                'name': ['bottom_right']
                            }, crs=all_black_segments.crs)  # Use the CRS of your tile polygon
                        elif i == max(range(0, width, tile_size)):
                            corners_gdf = gpd.GeoDataFrame({
                                'geometry': [bottom_left_polygon],
                                'name': ['bottom_left']
                            }, crs=all_black_segments.crs)
                        elif j == max(range(0, height, tile_size)):
                            corners_gdf = gpd.GeoDataFrame()
                        else:
                            corners_gdf = gpd.GeoDataFrame({
                                'geometry': [bottom_left_polygon, bottom_right_polygon],
                                'name': ['bottom_left', 'bottom_right']
                            }, crs=all_black_segments.crs)

                        # Write the corners to a GeoPackage
                        if not corners_gdf.empty:
                            corners_gpkg_filename = mask_output_filename.replace(".tif", "_corners.gpkg")
                            corners_gdf.to_file(corners_gpkg_filename, driver="GPKG")

                            # Convert the polygons to rasterized masks and apply them
                            corner_geometries = [(bottom_left_polygon, 1), (bottom_right_polygon, 1)]

                            corner_mask = rasterize(
                                corner_geometries,
                                out_shape=mask.shape,
                                transform=mask_transform,
                                fill=0,
                                default_value=1,
                                dtype=rasterio.uint8
                            )

                            # Apply the corner mask to the white tile mask
                            mask[corner_mask == 1] = 0

                    else:
                        print(f"No valid geometries found for tile ({i}, {j}).")
                else:
                    print(f"No overlapping black segments found for tile ({i}, {j}).")

                segmask_output_filename = mask_output_filename.replace(".tif", "_segmask.tif")

                with rasterio.open(mask_output_filename) as src:
                    profile = src.profile

                with rasterio.open(segmask_output_filename, 'w', **profile) as dst:
                    dst.write(mask.astype(rasterio.uint8), 1)

                #########################

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

                    bbox = segmented_image.total_bounds  # (minx, miny, maxx, maxy)
                    tile_boundary = box(bbox[0], bbox[1], bbox[2], bbox[3])

                    # Remove segments that intersect with the tile boundary
                    segmented_image_filtered = segmented_image[~segmented_image.intersects(tile_boundary.boundary)]

                    # Save the filtered segmented image for the white tile
                    segmented_output_filename = os.path.join(output_dir, f"segments_white_{i}_{j}_tile.gpkg")
                    segmented_image_filtered.to_file(segmented_output_filename, driver="GPKG")

                    all_white_segments = pd.concat([all_white_segments, segmented_image_filtered], ignore_index=True)
                except ValueError:
                    print(f"empty tile: {image_output_filename}")

            tile_index += 1
        tile_index += 1

    join_segments(output_dir, os.path.join(output_dir, "segments.gpkg"))


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
            mask, _ = open_binary_geotiff_as_mask(mask_output_filename)

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

                # Save the filtered segmented image with appropriate name
                segmented_output_filename = os.path.join(output_dir, f"segments_{j}_{i}_tile.gpkg")
                segmented_image.to_file(segmented_output_filename, driver="GPKG")
            except ValueError:
                print(f"empty tile: {image_output_filename}")

    join_segments(output_dir, os.path.join(output_dir, "segments.gpkg"))