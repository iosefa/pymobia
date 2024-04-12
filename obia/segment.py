import warnings
import numpy as np

from rasterio.features import shapes
from shapely.geometry import shape
from collections import defaultdict
from PIL.Image import fromarray
from pandas import Series
from geopandas import GeoDataFrame
from scipy import stats
from shapely.affinity import affine_transform
from skimage.segmentation import quickshift, slic, mark_boundaries
from skimage.util import img_as_float
from tqdm import tqdm


class SegmentsFrame(GeoDataFrame):
    _metadata = ['required_columns']

    required_columns = {
        'segment_id': float,
        'nobs': float,
        'b1_min': float,
        'b1_max': float,
        'b2_min': float,
        'b2_max': float,
        'b3_min': float,
        'b3_max': float,
        'b1_mean': float,
        'b2_mean': float,
        'b3_mean': float,
        'b1_variance': float,
        'b2_variance': float,
        'b3_variance': float,
        'b1_skewness': float,
        'b2_skewness': float,
        'b3_skewness': float,
        'b1_kurtosis': float,
        'b2_kurtosis': float,
        'b3_kurtosis': float,
        'geometry': any
    }

    def __init__(self, *args, **kwargs):
        super(SegmentsFrame, self).__init__(*args, **kwargs)

        if self.empty:
            for column, dtype in self.required_columns.items():
                self[column] = Series(dtype=dtype)
        else:
            self._validate_columns()

    def _validate_columns(self):
        for column in self.required_columns:
            if column not in self.columns:
                raise ValueError(f"Missing required column: {column}")


class ImageSegments:
    segments = None
    statistics = None
    img = None
    original_image = None
    method = None
    params = {}

    def __init__(self, img, method, **kwargs):
        self.original_image = img
        self.method = method
        self.params.update(kwargs)
        self._segment_image(img, method, **kwargs)
        self._create_segment_statistics(img)
        self._create_segmented_img(img)

    @staticmethod
    def _summary_statistics(segment_pixels):
        features = []
        n_pixels = segment_pixels.shape
        with warnings.catch_warnings(record=True):
            statistics = stats.describe(segment_pixels)
        band_stats = list(statistics)
        if n_pixels == 1:
            band_stats[3] = 0.0
        features += band_stats
        return features

    def _segment_image(self, image, method='quickshift', **kwargs):
        img = np.array(image.img)
        img = img_as_float(img)
        if method == 'quickshift':
            self.segments = quickshift(img, **kwargs)
        elif method == 'slic':
            self.segments = slic(img, **kwargs)
        else:
            raise Exception('An unknown segmentation method was requested.')

    def _create_segment_statistics(self, image):
        img = np.array(image.img)
        img = img_as_float(img)
        segment_ids = np.unique(self.segments)

        segment_stats = defaultdict(list)

        for segment_id in tqdm(segment_ids, bar_format='{l_bar}{bar}', desc="Analyzing Segments"):
            segment_mask = self.segments == segment_id
            segment_pixels = img[segment_mask]

            stats_dict = {
                'segment_id': segment_id,
                'nobs': np.nan,
                'b1_min': np.nan,
                'b1_max': np.nan,
                'b2_min': np.nan,
                'b2_max': np.nan,
                'b3_min': np.nan,
                'b3_max': np.nan,
                'b1_mean': np.nan,
                'b2_mean': np.nan,
                'b3_mean': np.nan,
                'b1_variance': np.nan,
                'b2_variance': np.nan,
                'b3_variance': np.nan,
                'b1_skewness': np.nan,
                'b2_skewness': np.nan,
                'b3_skewness': np.nan,
                'b1_kurtosis': np.nan,
                'b2_kurtosis': np.nan,
                'b3_kurtosis': np.nan,
                'feature_class': None,
                'geometry': None
            }

            mask = segment_mask.astype('int32')
            for s, v in shapes(mask):
                if v == 1:
                    geometry = shape(s)
                    transformed_geom = affine_transform(geometry, self.original_image.affine_transformation)
                    stats_dict['geometry'] = transformed_geom

            nobs = np.sum(segment_mask)
            stats_dict['nobs'] = nobs

            for band_index in range(3):
                band_stats = segment_pixels[:, band_index]
                band_prefix = f'b{band_index + 1}_'

                stats_dict[band_prefix + 'min'] = np.min(band_stats)
                stats_dict[band_prefix + 'max'] = np.max(band_stats)
                stats_dict[band_prefix + 'mean'] = np.mean(band_stats)
                stats_dict[band_prefix + 'variance'] = np.var(band_stats)
                stats_dict[band_prefix + 'skewness'] = stats.skew(band_stats, bias=False)
                stats_dict[band_prefix + 'kurtosis'] = stats.kurtosis(band_stats, bias=False)

            for key, value in stats_dict.items():
                segment_stats[key].append(value)

        self.statistics = SegmentsFrame(segment_stats)

    def _create_segmented_img(self, image):
        img = np.array(image.img)
        boundaries = mark_boundaries(img, self.segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        self.img = fromarray(masked_img.astype(np.uint8))

    def write_segments(self, file_path):
        self.statistics.to_file(file_path)
