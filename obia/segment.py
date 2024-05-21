import numpy as np
import pyproj
from numpy import ma

from rasterio.features import shapes
from shapely.geometry import shape
from collections import defaultdict
from PIL.Image import fromarray
from PIL.Image import Image as PILImage
from geopandas import GeoDataFrame
from scipy import stats
from shapely.affinity import affine_transform
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import quickshift, slic, mark_boundaries
from skimage.util import img_as_float
from tqdm import tqdm
from multiprocessing import Pool


def compute_stats(segment_id, segments, img_to_segment, statistics_bands, image, calc_mean,
                  calc_variance, calc_skewness, calc_kurtosis, calc_contrast, calc_dissimilarity,
                  calc_homogeneity, calc_ASM, calc_energy, calc_correlation):
    segment_mask = segments == segment_id
    segment_pixels = img_to_segment[segment_mask]

    stats_dict = {
        'segment_id': segment_id,
        'feature_class': None
    }

    bands = ['b' + f'{idx}' for idx in statistics_bands]

    operations = ['calc_mean', 'calc_variance', 'calc_skewness', 'calc_kurtosis',
                  'calc_contrast', 'calc_dissimilarity',
                  'calc_homogeneity', 'calc_ASM', 'calc_energy', 'calc_correlation']

    for band in bands:
        for op in operations:
            if locals()[op]:
                key = f'{band}_{op.split('_')[1]}'
                stats_dict[key] = np.nan

    mask = segment_mask.astype('int32')

    for band_index, band_prefix in enumerate(bands):
        mask_2d = segment_mask & (np.isfinite(image.img_data[:, :, band_index]))
        band_stats = np.where(mask_2d, image.img_data[:, :, band_index], np.nan)
        band_stats = ma.masked_invalid(band_stats)

        band_stats_no_nan = np.nan_to_num(band_stats.filled(0)).astype(np.uint8)
        GLCM = graycomatrix(band_stats_no_nan, distances=[5], angles=[0], levels=256, symmetric=False, normed=False)
        band_flat = ma.compressed(band_stats)

        if calc_mean:
            stats_dict[band_prefix + '_mean'] = np.mean(band_flat)
        if calc_variance:
            stats_dict[band_prefix + '_variance'] = np.var(band_flat)
        if calc_skewness:
            stats_dict[band_prefix + '_skewness'] = stats.skew(band_flat, bias=False)
        if calc_kurtosis:
            stats_dict[band_prefix + '_kurtosis'] = stats.kurtosis(band_flat, bias=False)
        if calc_contrast:
            props = graycoprops(GLCM, 'contrast')
            stats_dict[band_prefix + '_contrast'] = np.mean(props.flatten())
        if calc_dissimilarity:
            props = graycoprops(GLCM, 'dissimilarity')
            stats_dict[band_prefix + '_dissimilarity'] = np.mean(props.flatten())
        if calc_homogeneity:
            props = graycoprops(GLCM, 'homogeneity')
            stats_dict[band_prefix + '_homogeneity'] = np.mean(props.flatten())
        if calc_ASM:
            props = graycoprops(GLCM, 'ASM')
            stats_dict[band_prefix + '_ASM'] = np.mean(props.flatten())
        if calc_energy:
            props = graycoprops(GLCM, 'energy')
            stats_dict[band_prefix + '_energy'] = np.mean(props.flatten())
        if calc_correlation:
            props = graycoprops(GLCM, 'correlation')
            stats_dict[band_prefix + '_correlation'] = np.mean(props.flatten())

    for s, v in shapes(mask):
        if v == 1:
            geometry = shape(s)
            transformed_geom = affine_transform(geometry, image.affine_transformation)
            stats_dict['geometry'] = transformed_geom

    return stats_dict


class Segments:
    _segments = None
    segments = None
    method = None
    params = {}

    def __init__(self, _segments, segments, method, **kwargs):
        self._segments = _segments
        self.segments = segments
        self.method = method
        self.params.update(kwargs)

    def to_segmented_image(self, image):
        if not isinstance(image, PILImage):
            raise TypeError('Input must be a PIL Image')
        img = np.array(image)
        boundaries = mark_boundaries(img, self._segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        return fromarray(masked_img.astype(np.uint8))

    def write_segments(self, file_path):
        self.segments.to_file(file_path)


def compute_segment_stats_wrapper(args):
    return compute_stats(*args)


def segment(image, segmentation_bands=None, statistics_bands=None,
            method="slic", calc_mean=True, calc_variance=True,
            calc_skewness=True, calc_kurtosis=True, calc_contrast=True,
            calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True,
            calc_energy=True, calc_correlation=True, **kwargs):

    if segmentation_bands is None:
        segmentation_bands = [0, 1, 2]

    if not isinstance(segmentation_bands, (list, tuple)) or len(segmentation_bands) not in (1, 3):
        raise ValueError("'bands' should be a list or tuple of exactly one or three elements")

    num_bands = image.img_data.shape[2]

    if statistics_bands is None:
        statistics_bands = list(range(num_bands))

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

    segment_ids = np.unique(segments)

    segment_stats = defaultdict(list)

    with Pool(processes=16) as pool:
        args = ((seg_id, segments, img_to_segment, statistics_bands, image, calc_mean,
                 calc_variance, calc_skewness, calc_kurtosis, calc_contrast, calc_dissimilarity,
                 calc_homogeneity, calc_ASM, calc_energy, calc_correlation) for seg_id in segment_ids)
        results = list(tqdm(pool.imap(compute_segment_stats_wrapper, args), total=len(segment_ids)))

    for result in results:
        for key, value in result.items():
            segment_stats[key].append(value)

    gdf = GeoDataFrame(segment_stats, geometry=segment_stats['geometry'])

    srs = pyproj.CRS(image.crs)
    srs_epsg = srs.to_epsg()
    gdf.crs = f"EPSG:{srs_epsg}"

    return Segments(segments, gdf, method, **kwargs)
