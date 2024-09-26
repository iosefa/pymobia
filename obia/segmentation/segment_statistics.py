import numpy as np
import geopandas as gpd
import pandas as pd

from numpy import ma
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_fhd, calculate_chm, calculate_pai

from obia.utils.utils import mask_image_with_polygon


def _create_empty_stats_gdf(spectral_bands, textural_bands, calc_mean, calc_variance, calc_min, calc_max,
                              calc_skewness, calc_kurtosis,
                              calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy,
                              calc_correlation,
                              calc_pai, calc_fhd, calc_ch, calc_mean_intensity, calc_variance_intensity):
    columns = ['segment_id']

    spectral_stats = {
        "mean": calc_mean,
        "variance": calc_variance,
        "min": calc_min,
        "max": calc_max,
        "skewness": calc_skewness,
        "kurtosis": calc_kurtosis
    }

    textural_stats = {
        "contrast": calc_contrast,
        "dissimilarity": calc_dissimilarity,
        "homogeneity": calc_homogeneity,
        "ASM": calc_ASM,
        "energy": calc_energy,
        "correlation": calc_correlation
    }

    for band_index in spectral_bands:
        for stat, is_calculated in spectral_stats.items():
            if is_calculated:
                columns.append(f"b{band_index}_{stat}")

    for band_index in textural_bands:
        for stat, is_calculated in textural_stats.items():
            if is_calculated:
                columns.append(f"b{band_index}_{stat}")

    pointcloud_stats = {
        "pai": calc_pai,
        "fhd": calc_fhd,
        "ch": calc_ch,
        "mean_intensity": calc_mean_intensity,
        "variance_intensity": calc_variance_intensity
    }

    for stat, is_calculated in pointcloud_stats.items():
        if is_calculated:
            columns.append(stat)

    columns.append('geometry')

    gdf = gpd.GeoDataFrame(columns=columns, geometry='geometry')

    return gdf


from scipy.stats import skew, kurtosis

def calculate_spectral_stats(
        image, statistics_bands,
        calc_mean=True, calc_variance=True, calc_min=True, calc_max=True, calc_skewness=True, calc_kurtosis=True
):
    stats_dict = {}
    bands = ['b' + f'{idx}' for idx in statistics_bands]

    for band_index, band_prefix in enumerate(bands):
        band_stats = image[:, :, band_index]
        band_stats = ma.masked_invalid(band_stats)
        band_flat = ma.compressed(band_stats)

        if band_flat.size == 0:
            if calc_mean:
                stats_dict[band_prefix + '_mean'] = np.nan
            if calc_variance:
                stats_dict[band_prefix + '_variance'] = np.nan
            if calc_min:
                stats_dict[band_prefix + '_min'] = np.nan
            if calc_max:
                stats_dict[band_prefix + '_max'] = np.nan
            if calc_skewness:
                stats_dict[band_prefix + '_skewness'] = np.nan
            if calc_kurtosis:
                stats_dict[band_prefix + '_kurtosis'] = np.nan
        else:
            if calc_mean:
                stats_dict[band_prefix + '_mean'] = np.mean(band_flat)
            if calc_variance:
                stats_dict[band_prefix + '_variance'] = np.var(band_flat)
            if calc_min:
                stats_dict[band_prefix + '_min'] = np.min(band_flat)
            if calc_max:
                stats_dict[band_prefix + '_max'] = np.max(band_flat)
            if calc_skewness:
                stats_dict[band_prefix + '_skewness'] = skew(band_flat)
            if calc_kurtosis:
                stats_dict[band_prefix + '_kurtosis'] = kurtosis(band_flat)
    return stats_dict



def calculate_textural_stats(
        image, textural_bands,
        calc_contrast=True, calc_dissimilarity=True, calc_homogeneity=True,
        calc_ASM=True, calc_energy=True, calc_correlation=True
):
    stats_dict = {}

    bands = ['b' + f'{idx}' for idx in textural_bands]

    for band_index, band_prefix in enumerate(bands):
        band_stats = image[:, :, band_index]
        band_stats = ma.masked_invalid(band_stats)

        if band_stats.size == 0 or np.all(np.isnan(band_stats)):
            if calc_contrast:
                stats_dict[band_prefix + '_contrast'] = np.nan
            if calc_dissimilarity:
                stats_dict[band_prefix + '_dissimilarity'] = np.nan
            if calc_homogeneity:
                stats_dict[band_prefix + '_homogeneity'] = np.nan
            if calc_ASM:
                stats_dict[band_prefix + '_ASM'] = np.nan
            if calc_energy:
                stats_dict[band_prefix + '_energy'] = np.nan
            if calc_correlation:
                stats_dict[band_prefix + '_correlation'] = np.nan
        else:
            band_stats_no_nan = np.nan_to_num(band_stats.filled(0)).astype(np.uint8)

            glcm = graycomatrix(band_stats_no_nan, distances=[5], angles=[0], levels=256, symmetric=False, normed=True)

            if calc_contrast:
                stats_dict[band_prefix + '_contrast'] = np.mean(graycoprops(glcm, 'contrast').flatten())
            if calc_dissimilarity:
                stats_dict[band_prefix + '_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity').flatten())
            if calc_homogeneity:
                stats_dict[band_prefix + '_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity').flatten())
            if calc_ASM:
                stats_dict[band_prefix + '_ASM'] = np.mean(graycoprops(glcm, 'ASM').flatten())
            if calc_energy:
                stats_dict[band_prefix + '_energy'] = np.mean(graycoprops(glcm, 'energy').flatten())
            if calc_correlation:
                stats_dict[band_prefix + '_correlation'] = np.mean(graycoprops(glcm, 'correlation').flatten())

    return stats_dict


def calculate_structural_stats(
        pointcloud, voxel_resolution, calc_pai=True, calc_fhd=True, calc_ch=True
):
    stats_dict = {}
    voxels, extent = assign_voxels(pointcloud, voxel_resolution)
    if calc_pai:
        pad = calculate_pad(voxels, voxel_resolution[-1])
        pai = calculate_pai(pad)
        stats_dict['pai'] = np.mean(pai)
    if calc_fhd:
        fhd = calculate_fhd(voxels)
        stats_dict['fhd'] = np.mean(fhd)
    if calc_ch:
        ch, extent = calculate_chm(pointcloud, voxel_resolution)
        stats_dict['ch'] = np.mean(ch)
    return stats_dict


def calculate_radiometric_stats(
        pointcloud, calc_mean_intensity=True, calc_variance_intensity=True
):
    stats_dict = {}
    intensities = pointcloud['Intensity']
    if calc_mean_intensity:
        stats_dict['mean_intensity'] = np.mean(intensities)
    if calc_variance_intensity:
        stats_dict['variance_intensity'] = np.var(intensities)
    return stats_dict


def create_objects(
        segments, image, ept=None, ept_srs=None, spectral_bands=None, textural_bands=None, voxel_resolution=None,
        calculate_spectral=True, calculate_textural=True, calculate_structural=False, calculate_radiometric=False,
        calc_mean=True, calc_variance=True, calc_min=True, calc_max=True, calc_skewness=True, calc_kurtosis=True,
        calc_contrast=True, calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True, calc_energy=True, calc_correlation=True,
        calc_pai=True, calc_fhd=True, calc_ch=True, calc_mean_intensity=True, calc_variance_intensity=True
):
    if not calculate_spectral and not calculate_textural:
        raise ValueError("At least one of 'calculate_spectral' or 'calculate_textural' must be True.")

    if spectral_bands is None:
        spectral_bands = list(range(image.img_data.shape[2]))
    if textural_bands is None:
        textural_bands = list(range(image.img_data.shape[2]))

    gdf = _create_empty_stats_gdf(
        spectral_bands, textural_bands,
        calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis,
        calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy, calc_correlation,
        calc_pai, calc_fhd, calc_ch, calc_mean_intensity, calc_variance_intensity
    )

    for idx, segment in tqdm(segments.iterrows(), total=len(segments)):
        geom = segment.geometry
        masked_img_data = mask_image_with_polygon(image, geom)
        segment_id = segment['segment_id']
        row = {
            'segment_id': segment_id,
            'geometry': geom
        }

        spectral_statistics = calculate_spectral_stats(
            masked_img_data, spectral_bands,
            calc_mean=calc_mean, calc_variance=calc_variance, calc_min=calc_min,
            calc_max=calc_max, calc_skewness=calc_skewness, calc_kurtosis=calc_kurtosis
        )
        row.update(spectral_statistics)

        textural_statistics = calculate_textural_stats(
            masked_img_data, textural_bands,
            calc_contrast=calc_contrast, calc_dissimilarity=calc_dissimilarity, calc_homogeneity=calc_homogeneity,
            calc_ASM=calc_ASM, calc_energy=calc_energy, calc_correlation=calc_correlation
        )
        row.update(textural_statistics)

        if ept is not None:
            if ept_srs is None:
                raise ValueError("Error: 'ept_srs' must be provided when 'ept' is not None.")
            if voxel_resolution is None:
                raise ValueError("Error: 'voxel_resolution' must be provided when 'ept' is not None.")
            xmin, ymin, xmax, ymax = geom.bounds
            bounds = ([xmin, xmax], [ymin, ymax])

            pointclouds = read_lidar(ept, ept_srs, bounds, crop_poly=True, poly=geom.wkt)
            # todo: raise error if HeightAboveGround is missing.
            if calculate_structural:
                structural_statistics = calculate_structural_stats(
                    pointclouds[0], voxel_resolution,
                    calc_pai=calc_pai, calc_fhd=calc_fhd, calc_ch=calc_ch
                )
                row.update(structural_statistics)

            if calculate_radiometric:
                radiometric_statistics = calculate_radiometric_stats(
                    pointclouds[0],
                    calc_mean_intensity=calc_mean_intensity, calc_variance_intensity=calc_variance_intensity
                )
                row.update(radiometric_statistics)

        gdf = pd.concat([gdf, pd.DataFrame([row])], ignore_index=True)

    gdf.crs = segments.crs

    return gdf
