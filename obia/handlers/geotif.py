import numpy as np
from PIL.Image import fromarray
import rasterio


class Image:
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
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1).astype(bool)
    return mask_array
