import numpy as np
from PIL.Image import fromarray
from PIL.Image import Image as PILImage
from skimage.segmentation import mark_boundaries

from obia.segmentation.segment_boundaries import create_segments
from obia.segmentation.segment_statistics import create_objects


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


def segment(image, segmentation_bands=None, statistics_bands=None,
            method="slic", calc_mean=True, calc_variance=True,
            calc_skewness=True, calc_kurtosis=True, calc_contrast=True,
            calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True,
            calc_energy=True, calc_correlation=True, **kwargs):

    segments_gdf = create_segments(image, segmentation_bands=segmentation_bands, method=method, **kwargs)
    objects_gdf = create_objects(segments_gdf, image, statistics_bands=statistics_bands, calc_mean=calc_mean,
                                 calc_variance=calc_variance, calc_skewness=calc_skewness, calc_kurtosis=calc_kurtosis,
                                 calc_contrast=calc_contrast, calc_dissimilarity=calc_dissimilarity,
                                 calc_homogeneity=calc_homogeneity, calc_ASM=calc_ASM, calc_energy=calc_energy,
                                 calc_correlation=calc_correlation)

    return Segments(segments_gdf, objects_gdf, method, **kwargs)
