import numpy as np
from numpy import ma
from random import randint
from skimage.transform import rotate, SimilarityTransform, warp
from scipy.ndimage import center_of_mass
from skimage.filters import gaussian


def _histogram_matching(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image

    Source: https://stackoverflow.com/a/33047048/604734 (Creative Commons)
    """

    template = template.copy()
    source = source.copy()
    template[template.mask] = 0

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    t_values = np.delete(t_values, 0)
    t_counts = np.delete(t_counts, 0)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def mix_lesions(lesion_bg, lesion_fg, mask_bg, mask_fg, gauss_sigma=0):
    height, width = lesion_bg.shape[:2]

    # Histogram matching
    for i in range(3):
        lesion_bg_masked = ma.array(lesion_bg[..., i], mask=~mask_bg)
        lesion_fg[..., i] = _histogram_matching(lesion_fg[..., i],
                                                lesion_bg_masked)

    rotation = randint(0, 90)
    lesion_fg = rotate(lesion_fg, rotation, mode='reflect',
                       preserve_range=True).astype('uint8')
    mask_fg = rotate(mask_fg, rotation, mode='constant', cval=0,
                     preserve_range=True).astype('uint8')

    cm_fg = center_of_mass(mask_fg)
    cm_bg = center_of_mass(mask_bg)

    tf_ = SimilarityTransform(
        scale=1, rotation=0,
        translation=(cm_fg[1] - cm_bg[1], cm_fg[0] - cm_bg[0]))

    lesion_fg = warp(lesion_fg, tf_, mode='constant',
                     preserve_range=True).astype('uint8')
    mask_fg = warp(mask_fg, tf_, mode='constant', cval=0,
                   preserve_range=True).astype('uint8')
    cm_fg = center_of_mass(mask_fg)

    # Cut mask
    cut_mask = np.zeros(mask_fg.shape)
    cut_mask[cut_mask.shape[0] // 2:, :] = 255
    cut_mask = rotate(cut_mask,
                      randint(0, 90),
                      mode='reflect',
                      preserve_range=True).astype('uint8')
    tf_cm = SimilarityTransform(
        scale=1,
        rotation=0,
        translation=(width // 2 - cm_bg[1], height // 2 - cm_bg[0]))
    cut_mask = warp(cut_mask, tf_cm, mode='constant', cval=0,
                    preserve_range=True).astype('uint8')
    mask_fg = np.where(np.logical_and(mask_fg, cut_mask), 255, 0)

    # Calculate mask bounding box
    coords = np.argwhere(mask_fg == 255)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Convert mask to 3 channels
    mask_fg = np.dstack((mask_fg, mask_fg, mask_fg))
    # Convert it to float
    mask_fg = mask_fg.astype('float')
    # And normalize it to 0.0~1.0
    mask_fg *= (1.0/255.0)

    # Apply Gaussian Blur to the mask
    mask_fg = gaussian(mask_fg, sigma=gauss_sigma, multichannel=True,
                       preserve_range=True)

    out = np.copy(lesion_bg)

    out_ = (lesion_bg * (1.0 - mask_fg) + lesion_fg * mask_fg).astype('uint8')
    out = np.where(mask_fg == 0, lesion_bg, out_)

    return out
