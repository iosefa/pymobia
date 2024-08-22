import numpy as np
import pyproj
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
from shapely.affinity import affine_transform
from skimage.segmentation import quickshift, slic
from skimage.util import img_as_float


def create_segments(image, segmentation_bands=None, method="slic", **kwargs):
    if segmentation_bands is None:
        segmentation_bands = [0, 1, 2]

    num_bands = image.img_data.shape[2]

    for band in segmentation_bands:
        if band >= num_bands or band < 0:
            raise IndexError(f"Band index {band} out of range. Available bands indices: 0 to {num_bands - 1}.")

    if len(segmentation_bands) == 3:
        img_to_segment = np.empty((image.img_data.shape[0], image.img_data.shape[1], 3))
        for i, band in enumerate(segmentation_bands):
            img_to_segment[:, :, i] = img_as_float(image.img_data[:, :, band])
    else:
        img_to_segment = img_as_float(image.img_data[:, :, segmentation_bands[0]])

    if method == 'quickshift':
        segments = quickshift(img_to_segment, **kwargs)
    elif method == 'slic':
        segments = slic(img_to_segment, **kwargs)
    else:
        raise Exception('An unknown segmentation method was requested.')

    mask = kwargs.get('mask', None)
    if mask is not None:
        segments[mask == 0] = -1

    segment_ids = np.unique(segments)

    geometries = []
    for segment_id in segment_ids:
        if segment_id == -1:
            continue
        segment_mask = segments == segment_id
        for s, v in shapes(segment_mask.astype('int32')):
            if v == 1:
                geometry = shape(s)
                transformed_geom = affine_transform(geometry, image.affine_transformation)
                geometries.append(transformed_geom)

    gdf = GeoDataFrame(geometry=geometries)

    srs = pyproj.CRS(image.crs)
    srs_epsg = srs.to_epsg()
    gdf.crs = f"EPSG:{srs_epsg}"

    return gdf
