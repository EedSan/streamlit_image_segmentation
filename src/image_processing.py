import numpy as np
import rasterio
from rasterio.enums import Resampling
import skimage.transform as st


def check_image_size(img_, resolution_=(256, 256)):
    if img_.size != resolution_:
        img_ = image_thumbnailing(img_, resolution_)
    return img_


def image_thumbnailing(file_, resolution_=(256, 256)):
    with rasterio.io.MemoryFile(file_) as mem:
        with mem.open() as src:
            aspect_ratio = src.width / src.height
            if aspect_ratio >= 1:
                new_height = int(resolution_[0] / aspect_ratio)
                new_width = resolution_[1]
            else:
                new_height = resolution_[0]
                new_width = int(resolution_[1] * aspect_ratio)
            data = src.read(out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear)
            transform = src.transform * src.transform.scale((src.width / data.shape[-1]), (src.height / data.shape[-2]))

    output = rasterio.io.MemoryFile()
    with output.open(driver='GTiff', height=new_height, width=new_width, count=src.count, dtype=data.dtype,
                     crs=src.crs, transform=transform, ) as dest:
        dest.write(data)
    return output


def is_geospatial(img_data_):
    return img_data_.crs is not None and img_data_.transform is not None


def get_img_762bands_from_mem_file(img_):
    with rasterio.io.MemoryFile(img_) as mem_file:
        with mem_file.open() as img:
            is_geo = is_geospatial(img)
            try:
                img = img.read((7, 6, 2)).transpose((1, 2, 0))
                bands_msg_ = 'Image is processed successfully. '
            except Exception:
                bands_msg_ = 'Image Exception: Amount of image bands is less than 10. '
                img = img.read().transpose((1, 2, 0))
            img = np.float32(img) / 65535
    bands_msg_ += "| Is geoimage: " + str(is_geo)
    return img, bands_msg_


def get_img_762bands_from_path(path_):
    img_ = rasterio.open(path_).read((7, 6, 2)).transpose((1, 2, 0))
    img_ = np.float32(img_) / 65535
    return img_


def image_processing(img_):
    img_ = check_image_size(img_)
    img_, bands_msg_ = get_img_762bands_from_mem_file(img_)
    return img_, bands_msg_
