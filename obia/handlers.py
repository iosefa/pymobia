import numpy as np
from PIL.Image import fromarray
from affine import Affine
from osgeo import gdal, osr
import rasterio
from skimage import exposure


class Image:
    img = None
    crs = None
    transform = None
    affine_transformation = None

    def __init__(self, img, crs, affine_transformation, transform):
        self.img = img
        self.crs = crs
        self.affine_transformation = affine_transformation
        self.transform = transform


def open_geotiff(image, chm=None):
    raster = gdal.Open(image, gdal.GA_ReadOnly)
    projection = raster.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    crs = srs.ExportToProj4()

    gt = raster.GetGeoTransform()
    transform = Affine.from_gdal(*gt)
    affine_transformation = [gt[1], gt[2], gt[4], gt[5], gt[0], gt[3]]

    n_bands = raster.RasterCount
    bands_data = []

    for b in range(1, n_bands + 1):
        band = raster.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack([b for b in bands_data])

    x_coords, y_coords = np.meshgrid(np.arange(bands_data.shape[1]), np.arange(bands_data.shape[0]))

    if chm is not None:
        chm_raster = gdal.Open(chm, gdal.GA_ReadOnly)
        chm_projection = chm_raster.GetProjection()
        chm_srs = osr.SpatialReference(wkt=chm_projection)
        if chm_srs != srs:
            raise Exception("raster image srs and chm srs do not match")

        z_coords = chm_raster[:, :, 0]


    spatial_color_image = np.dstack((x_coords_normalized, y_coords_normalized, normalized_dsm, normalized_img))

    red = exposure.rescale_intensity(bands_data[:, :, 0])
    green = exposure.rescale_intensity(bands_data[:, :, 1])
    blue = exposure.rescale_intensity(bands_data[:, :, 2])

    img = np.dstack((red, green, blue))
    img = img.astype(np.float32)
    normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = fromarray((normalized_img * 255).astype(np.uint8))
    return Image(img, crs, affine_transformation, transform)


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
