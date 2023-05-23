import PIL

# import os
from PIL import Image

def image_resize(width, height, image):
    # os.makedirs(out_dir, exist_ok=True)
    # new_image = Image.open(image_file)
    resized_image = image.resize((width, height),resample=Image.Resampling.NEAREST)
    return resized_image
    # filename = os.path.basename(image_file)
    # save_path = os.path.join(out_dir, f'{filename}')
    # resized_image.save(save_path)


# Example usage:
#image_resize(250, 250, 'raw_data/OpenEarthMap_wo_xBD/aachen/images/aachen_1.tif', 'path/to/output/directory')
