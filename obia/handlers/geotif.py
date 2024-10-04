import numpy as np
from PIL.Image import fromarray
import rasterio


class Image:
    """
        Image
        A class for handling geographic raster image data, providing utilities to manage its properties and to convert it into visualization formats.

        Attributes
        ----------
        img_data : numpy.ndarray
            The image data array.
        crs : str or dict
            Coordinate reference system of the image.
        transform : affine.Affine
            Affine transformation matrix.
        affine_transformation : affine.Affine
            Deprecated attribute for affine transformation.
        rasterio_obj : rasterio.io.DatasetReader
            Rasterio dataset object for the image.

        Methods
        -------
        __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj)
            Constructs all the necessary attributes for the Image object.
        to_image(self, bands)
            Converts specified bands of the image into an RGB image.
    """
    img_data = None
    crs = None
    transform = None
    affine_transformation = None
    rasterio_obj = None

    def __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj):
        self.img_data = img_data
        self.crs = crs
        self.affine_transformation = affine_transformation
        self.transform = transform
        self.rasterio_obj = rasterio_obj

    def to_image(self, bands):
        """
        :param bands: A list or tuple of three integers representing the indices of the bands to use for the RGB image.
        :return: An RGB image created from the specified bands.
        """
        if not isinstance(bands, (list, tuple)) or len(bands) != 3:
            raise ValueError("'bands' should be a list or tuple of exactly three elements")

        rgb_data = np.empty((self.img_data.shape[0], self.img_data.shape[1], 3), dtype=np.uint8)

        num_bands = self.img_data.shape[2]

        for i, band in enumerate(bands):
            if band >= num_bands or band < 0:
                raise IndexError(f"Band index {band} out of range. Available bands indices: 0 to {num_bands - 1}.")
            rgb_data[:, :, i] = self.img_data[:, :, band]

        return fromarray(rgb_data)


def open_geotiff(image_path, bands=None):
    """
    :param image_path: Path to the GeoTIFF file to be opened.
    :type image_path: str
    :param bands: List of band indices to be read from the GeoTIFF file. If None, all bands are read.
    :type bands: list of int, optional
    :return: An Image object containing the raster data, coordinate reference system, affine transformation matrix, and the original rasterio object.
    :rtype: Image
    """
    rasterio_obj = rasterio.open(image_path)

    crs = rasterio_obj.crs
    transform = rasterio_obj.transform
    affine_transformation = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]

    x_size = rasterio_obj.width
    y_size = rasterio_obj.height
    num_bands = rasterio_obj.count

    if bands is None:
        bands = list(range(1, num_bands + 1))

    data = np.empty((y_size, x_size, len(bands)), dtype=np.float32)

    for i, b in enumerate(bands):
        band_array = rasterio_obj.read(b)
        data[:, :, i] = band_array

    return Image(data, crs, affine_transformation, transform, rasterio_obj)


def _write_geotiff(pil_image, output_path, crs, transform):
    data = np.array(pil_image)
    data = data.astype(np.uint8)

    bands, height, width = data.shape if len(data.shape) == 3 else (1, *data.shape)

    new_image = rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform)
    if bands > 1:
        for i in range(bands):
            new_image.write(data[i], i + 1)
    else:
        new_image.write(data, 1)
    new_image.close()
    print(f"Done Writing GeoTIFF at {output_path}")


def open_binary_geotiff_as_mask(mask_path):
    """
    :param mask_path: Path to the binary GeoTIFF file to be opened.
    :return: A tuple containing the binary mask array, bounding box, affine transform, and profile of the raster.
    """
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1).astype(bool)  # Read the mask as a binary array
        transform = src.transform  # Get the affine transform of the raster
        width, height = src.width, src.height  # Get the dimensions of the raster
        profile = src.profile  # Capture the profile to use later

        # Calculate the bounding box from the transform and dimensions
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        bbox = (left, bottom, right, top)

    return mask_array, bbox, transform, profile
