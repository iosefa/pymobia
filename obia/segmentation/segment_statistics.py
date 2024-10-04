import numpy as np
import geopandas as gpd
import pandas as pd

from numpy import ma
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_fhd, calculate_chm, calculate_pai

from obia.utils.utils import mask_image_with_polygon, crop_image_to_bbox


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
    """
    :param image: Input 3D numpy array where each band is along the third dimension.
    :param statistics_bands: List of band indices for which statistics need to be calculated.
    :param calc_mean: Boolean flag to calculate mean of the bands.
    :param calc_variance: Boolean flag to calculate variance of the bands.
    :param calc_min: Boolean flag to calculate minimum value of the bands.
    :param calc_max: Boolean flag to calculate maximum value of the bands.
    :param calc_skewness: Boolean flag to calculate skewness of the bands.
    :param calc_kurtosis: Boolean flag to calculate kurtosis of the bands.
    :return: Dictionary containing the calculated statistics for each specified band.
    """
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
    """
    :param image: The input image as a 3D numpy array where each band represents a separate textural feature.
    :param textural_bands: A list of indices specifying which bands to consider for textural analysis.
    :param calc_contrast: Boolean flag indicating whether to calculate the contrast metric.
    :param calc_dissimilarity: Boolean flag indicating whether to calculate the dissimilarity metric.
    :param calc_homogeneity: Boolean flag indicating whether to calculate the homogeneity metric.
    :param calc_ASM: Boolean flag indicating whether to calculate the Angular Second Moment (ASM) metric.
    :param calc_energy: Boolean flag indicating whether to calculate the energy metric.
    :param calc_correlation: Boolean flag indicating whether to calculate the correlation metric.
    :return: A dictionary containing the computed textural statistics for each specified band.
    """
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
            band_stats_no_nan = np.ma.masked_invalid(band_stats)
            if band_stats_no_nan.dtype == np.uint8:
                glcm_input = band_stats_no_nan
            else:
                band_min, band_max = np.min(band_stats_no_nan), np.max(band_stats_no_nan)
                glcm_input = ((band_stats_no_nan - band_min) / (band_max - band_min) * 255).astype(np.uint8)

            glcm = graycomatrix(glcm_input, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)

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
    """
    :param pointcloud: Input point cloud data.
    :param voxel_resolution: Resolution for voxel space.
    :param calc_pai: Flag to calculate Projected Area Index (PAI).
    :param calc_fhd: Flag to calculate Foliage Height Diversity (FHD).
    :param calc_ch: Flag to calculate Canopy Height (CH).
    :return: Dictionary containing calculated structural statistics.
    """
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
    """
    :param pointcloud: A dictionary containing point cloud data from which intensity values can be extracted.
    :param calc_mean_intensity: A boolean flag to determine whether the mean intensity should be calculated.
    :param calc_variance_intensity: A boolean flag to determine whether the variance of intensity should be calculated.
    :return: A dictionary containing the calculated mean and/or variance of intensity, depending on the flags provided.
    """
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
    """
    :param segments: GeoDataFrame containing the segmented regions to be analyzed.
    :param image: Object containing image data and metadata to be used for analysis.
    :param ept: Optional; Path to the EPT (Entwine Point Tiles) point cloud data. Defaults to None.
    :param ept_srs: Optional; Spatial reference system for the EPT data. Required if ept is provided. Defaults to None.
    :param spectral_bands: Optional; List of spectral bands to be used in the analysis. Defaults to all available bands.
    :param textural_bands: Optional; List of textural bands to be used in the analysis. Defaults to all available bands.
    :param voxel_resolution: Optional; Voxel resolution for 3D point cloud data analysis. Required if ept is provided. Defaults to None.
    :param calculate_spectral: Boolean; Whether to calculate spectral statistics. Defaults to True.
    :param calculate_textural: Boolean; Whether to calculate textural statistics. Defaults to True.
    :param calculate_structural: Boolean; Whether to calculate structural statistics using point cloud data. Defaults to False.
    :param calculate_radiometric: Boolean; Whether to calculate radiometric statistics using point cloud data. Defaults to False.
    :param calc_mean: Boolean; Whether to calculate the mean of the pixel values. Defaults to True.
    :param calc_variance: Boolean; Whether to calculate the variance of the pixel values. Defaults to True.
    :param calc_min: Boolean; Whether to calculate the minimum of the pixel values. Defaults to True.
    :param calc_max: Boolean; Whether to calculate the maximum of the pixel values. Defaults to True.
    :param calc_skewness: Boolean; Whether to calculate the skewness of the pixel values. Defaults to True.
    :param calc_kurtosis: Boolean; Whether to calculate the kurtosis of the pixel values. Defaults to True.
    :param calc_contrast: Boolean; Whether to calculate the contrast for textural analysis. Defaults to True.
    :param calc_dissimilarity: Boolean; Whether to calculate the dissimilarity for textural analysis. Defaults to True.
    :param calc_homogeneity: Boolean; Whether to calculate the homogeneity for textural analysis. Defaults to True.
    :param calc_ASM: Boolean; Whether to calculate the Angular Second Moment (ASM) for textural analysis. Defaults to True.
    :param calc_energy: Boolean; Whether to calculate the energy for textural analysis. Defaults to True.
    :param calc_correlation: Boolean; Whether to calculate the correlation for textural analysis. Defaults to True.
    :param calc_pai: Boolean; Whether to calculate the Plant Area Index (PAI) from point cloud data. Defaults to True.
    :param calc_fhd: Boolean; Whether to calculate the Foliage Height Diversity (FHD) from point cloud data. Defaults to True.
    :param calc_ch: Boolean; Whether to calculate the Canopy Height (CH) from point cloud data. Defaults to True.
    :param calc_mean_intensity: Boolean; Whether to calculate the mean intensity from point cloud data. Defaults to True.
    :param calc_variance_intensity: Boolean; Whether to calculate the variance of intensity from point cloud data. Defaults to True.
    :return: GeoDataFrame containing the calculated statistics for each segment.
    """
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
        segment_id = segment['segment_id']

        cropped_img_data, cropped_transform = crop_image_to_bbox(image, geom)
        masked_img_data = mask_image_with_polygon(cropped_img_data, geom, cropped_transform)

        # masked_img_data = mask_image_with_polygon(image, geom)
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
        break

    gdf.crs = segments.crs

    return gdf
