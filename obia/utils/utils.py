import geopandas as gpd
from typing import List, Tuple

import numpy as np
from rasterio.features import geometry_mask


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


def mask_image_with_polygon(image, polygon):
    """Masks all pixels outside the polygon for the given image."""
    mask = geometry_mask([polygon], transform=image.transform, invert=True, out_shape=(image.img_data.shape[0], image.img_data.shape[1]))

    masked_img_data = np.where(mask[:, :, np.newaxis], image.img_data, np.nan)

    return masked_img_data
