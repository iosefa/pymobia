import numpy as np
from PIL.Image import fromarray
from affine import Affine
from osgeo import gdal, osr
import rasterio
from skimage import exposure


class Image:
    img_data = None
    crs = None
    transform = None
    affine_transformation = None

    def __init__(self, img_data, crs, affine_transformation, transform):
        self.img_data = img_data
        self.crs = crs
        self.affine_transformation = affine_transformation
        self.transform = transform

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
    raster = gdal.Open(image_path, gdal.GA_ReadOnly)

    projection = raster.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    crs = srs.ExportToProj4()
    gt = raster.GetGeoTransform()
    affine_transformation = [gt[1], gt[2], gt[4], gt[5], gt[0], gt[3]]
    transform = Affine.from_gdal(*gt)

    x_size = raster.RasterXSize
    y_size = raster.RasterYSize
    num_bands = raster.RasterCount

    if bands is None:
        bands = list(range(1, num_bands + 1))

    data = np.empty((y_size, x_size, len(bands)), dtype=np.float32)

    for i, b in enumerate(bands):
        band = raster.GetRasterBand(b)
        band_array = band.ReadAsArray()
        data[:, :, i] = band_array

    return Image(data, crs, affine_transformation, transform)


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
