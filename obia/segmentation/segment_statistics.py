import numpy as np
from numpy import ma
from rasterio.mask import mask
from shapely.geometry import box
from collections import defaultdict
from geopandas import GeoDataFrame
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


def compute_stats(segment_id, statistics_bands, image, mask, calc_mean,
                  calc_variance, calc_skewness, calc_kurtosis, calc_contrast, calc_dissimilarity,
                  calc_homogeneity, calc_ASM, calc_energy, calc_correlation):

    stats_dict = {
        'segment_id': segment_id,
    }

    bands = ['b' + f'{idx}' for idx in statistics_bands]

    for band_index, band_prefix in enumerate(bands):
        band_stats = np.where(mask, image[:, :, band_index], np.nan)
        band_stats = ma.masked_invalid(band_stats)
        band_flat = ma.compressed(band_stats)

        if calc_mean:
            stats_dict[band_prefix + '_mean'] = np.mean(band_flat)
        if calc_variance:
            stats_dict[band_prefix + '_variance'] = np.var(band_flat)
        if calc_skewness:
            stats_dict[band_prefix + '_skewness'] = stats.skew(band_flat, bias=False)
        if calc_kurtosis:
            stats_dict[band_prefix + '_kurtosis'] = stats.kurtosis(band_flat, bias=False)

        if calc_contrast or calc_dissimilarity or calc_homogeneity or calc_ASM or calc_energy or calc_correlation:
            band_stats_no_nan = np.nan_to_num(band_stats.filled(0)).astype(np.uint8)
            GLCM = graycomatrix(band_stats_no_nan, distances=[5], angles=[0], levels=256, symmetric=False, normed=False)

            if calc_contrast:
                stats_dict[band_prefix + '_contrast'] = np.mean(graycoprops(GLCM, 'contrast').flatten())
            if calc_dissimilarity:
                stats_dict[band_prefix + '_dissimilarity'] = np.mean(graycoprops(GLCM, 'dissimilarity').flatten())
            if calc_homogeneity:
                stats_dict[band_prefix + '_homogeneity'] = np.mean(graycoprops(GLCM, 'homogeneity').flatten())
            if calc_ASM:
                stats_dict[band_prefix + '_ASM'] = np.mean(graycoprops(GLCM, 'ASM').flatten())
            if calc_energy:
                stats_dict[band_prefix + '_energy'] = np.mean(graycoprops(GLCM, 'energy').flatten())
            if calc_correlation:
                stats_dict[band_prefix + '_correlation'] = np.mean(graycoprops(GLCM, 'correlation').flatten())

    return stats_dict


def create_objects(segments, image, statistics_bands=None, calc_mean=True, calc_variance=True,
                   calc_skewness=True, calc_kurtosis=True, calc_contrast=True, calc_dissimilarity=True,
                   calc_homogeneity=True, calc_ASM=True, calc_energy=True, calc_correlation=True):

    if statistics_bands is None:
        statistics_bands = list(range(image.img_data.shape[2]))

    segment_stats = defaultdict(list)

    segments['segment_id'] = range(1, len(segments) + 1)

    for idx, segment in tqdm(segments.iterrows(), total=len(segments)):
        geom = segment.geometry
        segment_id = segment['segment_id']

        geom_bbox = [box(*geom.bounds)]

        clipped_raster, clipped_transform = mask(image.rasterio_obj, geom_bbox, crop=True)

        if len(clipped_raster.shape) == 3:
            clipped_raster = clipped_raster.transpose((1, 2, 0))

        segment_mask = mask(image.rasterio_obj, [geom], crop=True, all_touched=True, invert=False)[0][0]

        if segment_mask.shape != clipped_raster.shape[:2]:
            continue

        stats = compute_stats(
            segment_id=segment_id,
            statistics_bands=statistics_bands,
            image=clipped_raster,
            mask=segment_mask,
            calc_mean=calc_mean,
            calc_variance=calc_variance,
            calc_skewness=calc_skewness,
            calc_kurtosis=calc_kurtosis,
            calc_contrast=calc_contrast,
            calc_dissimilarity=calc_dissimilarity,
            calc_homogeneity=calc_homogeneity,
            calc_ASM=calc_ASM,
            calc_energy=calc_energy,
            calc_correlation=calc_correlation
        )

        for key, value in stats.items():
            segment_stats[key].append(value)

    object_gdf = GeoDataFrame(segment_stats, geometry=segments.geometry)

    object_gdf.crs = segments.crs

    return object_gdf
