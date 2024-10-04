import geopandas as gpd
from typing import List, Tuple

import numpy as np
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


def label_segments(segments: gpd.GeoDataFrame, labelled_points: gpd.GeoDataFrame) -> Tuple[
    gpd.GeoDataFrame, List[str]]:
    """
    :param segments: A GeoDataFrame representing the segments to be labeled.
    :param labelled_points: A GeoDataFrame representing the labeled points used for segment labeling.
    :return: A tuple containing a GeoDataFrame with labeled segments and a list of segment IDs for mixed segments.
    """
    mixed_segments = []
    labelled_segments = segments.copy()
    intersections = gpd.sjoin(labelled_segments, labelled_points, how='inner', predicate='intersects')

    for polygon_id, group in intersections.groupby(intersections.index):
        classes = group['class'].unique()

        if len(classes) == 1:
            labelled_segments.loc[polygon_id, 'feature_class'] = classes[0]
        else:
            segment_id = group['segment_id'].values[0]
            mixed_segments.append(segment_id)

    labelled_segments = labelled_segments[labelled_segments['feature_class'].notna()]

    return labelled_segments, mixed_segments


def crop_image_to_bbox(image, geom):
    """
    Crop the image data to the bounding box of the given geometry.

    :param image: The Image object containing the image data and rasterio object.
    :param geom: The geometry (Polygon) used to derive the bounding box for cropping.
    :return: Cropped image data as a NumPy array and the updated transform.
    """
    xmin, ymin, xmax, ymax = geom.bounds
    window = from_bounds(xmin, ymin, xmax, ymax, transform=image.transform)
    cropped_img_data = image.rasterio_obj.read(window=window)
    cropped_transform = image.rasterio_obj.window_transform(window)

    return cropped_img_data, cropped_transform


def mask_image_with_polygon(cropped_img_data, polygon, cropped_transform):
    """
    Masks all pixels outside the polygon for the given cropped image.

    :param cropped_img_data: The cropped image data as a NumPy array.
    :param polygon: The geometry (Polygon) used for masking.
    :param cropped_transform: The affine transform for the cropped image.
    :return: Masked image data as a NumPy array.
    """
    height, width = cropped_img_data.shape[1], cropped_img_data.shape[2]
    mask = geometry_mask([polygon], transform=cropped_transform, invert=True, out_shape=(height, width))
    mask_expanded = np.expand_dims(mask, axis=0)  # Add an extra dimension for bands
    masked_img_data = np.where(mask_expanded, cropped_img_data, np.nan)

    return masked_img_data
