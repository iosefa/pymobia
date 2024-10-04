import numpy as np
import pyproj
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
from shapely.affinity import affine_transform
from skimage.segmentation import quickshift, slic
from skimage.util import img_as_float


def normalize_band(band):
    """
    :param band: The input band array that needs to be normalized. This should be a numpy array of numeric values.
    :return: A numpy array with the normalized values, where the minimum value is mapped to 0 and the maximum value is mapped to 1.
    """
    return (band - np.min(band)) / (np.max(band) - np.min(band))

def create_segments(image, segmentation_bands=None, method="slic", **kwargs):
    """
    :param image: The image to segment, composed of multiple bands.
    :type image: Image object
    :param segmentation_bands: List of band indices to use for segmentation. If None, all bands are used.
    :type segmentation_bands: list[int] or None
    :param method: Segmentation method to use: 'slic' or 'quickshift'.
    :type method: str
    :param kwargs: Additional keyword arguments passed to the segmentation algorithm.
    :type kwargs: dict
    :return: GeoDataFrame containing segmented geometries, with a column for segment IDs.
    :rtype: GeoDataFrame
    """
    num_bands = image.img_data.shape[2]
    for i in range(image.img_data.shape[2]):
        image.img_data[:, :, i] = normalize_band(image.img_data[:, :, i])
    # Use all bands if segmentation_bands is None
    if segmentation_bands is None:
        segmentation_bands = list(range(num_bands))

    for band in segmentation_bands:
        if band >= num_bands or band < 0:
            raise IndexError(f"Band index {band} out of range. Available bands indices: 0 to {num_bands - 1}.")

    # Prepare the image data for segmentation
    img_to_segment = img_as_float(image.img_data[:, :, segmentation_bands])

    # Check shape of img_to_segment before passing to slic
    print("Shape of img_to_segment:", img_to_segment.shape)

    if method == 'quickshift':
        segments = quickshift(img_to_segment, **kwargs)
    elif method == 'slic':
        segments = slic(img_to_segment, **kwargs)  # Ensure no channel_axis or conflicting arguments
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
    gdf['segment_id'] = range(1, len(gdf) + 1)
    return gdf
