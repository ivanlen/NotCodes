from PIL import Image


def square_pad_image(pil_im, desired_size=224):
    if isinstance(pil_im, Image.Image):
        im = pil_im
    elif isinstance(pil_im, str):
        im = Image.open(pil_im)
    else:
        raise Exception('invalid image type')
    old_size = im.size

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))
    return new_im


def center_crop_square(pil_im, desired_size=224):
    if isinstance(pil_im, Image.Image):
        im = pil_im.copy()
    elif isinstance(pil_im, str):
        im = Image.open(pil_im)
    else:
        raise Exception('invalid image type')
    new_width, new_height = [min(im.size)] * 2
    width, height = im.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.thumbnail([desired_size, desired_size], Image.ANTIALIAS)
    return im


def depreprocess(x):
    # https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/utils.py
    # https://github.com/tensorflow/tensorflow/blob/efb5a568e9de274840ae313c034c8ea70e222bbb/tensorflow/python/keras/applications/resnet.py#L519

    mean = [103.939, 116.779, 123.68]
    x1 = x[:, :, ::-1].copy()
    x1[..., 0] += mean[2]
    x1[..., 1] += mean[1]
    x1[..., 2] += mean[0]
    return x1  # [:,:,::-1]
