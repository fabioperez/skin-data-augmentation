from PIL import Image
from torchvision.transforms import Resize, TenCrop


def _inception_crop(img, size, resizes=(256, 288, 320, 352)):
    resized = [Resize(size_)(img) for size_ in resizes]
    results = []
    for resized_img in resized:
        w, h = resized_img.size
        if w < h:
            t1 = resized_img.crop((0, 0, w, w))
            t2 = resized_img.crop((0, h/2 - w/2, w, h/2 + w/2))
            t3 = resized_img.crop((0, h-w, w, h))
        else:
            t1 = resized_img.crop((0, 0, h, h))
            t2 = resized_img.crop((w/2 - h/2, 0, w/2 + h/2, h))
            t3 = resized_img.crop((w-h, 0, w, h))

        for square in (t1, t2, t3):
            resized_square = Resize(size)(square)
            results.append(resized_square)
            results.append(resized_square.transpose(Image.FLIP_LEFT_RIGHT))
            results += TenCrop(size)(square)

    return results


class InceptionCrop(object):
    def __init__(self, size, resizes):
        self.size = size
        self.resizes = resizes

    def __call__(self, img):
        return _inception_crop(img, self.size, self.resizes)
