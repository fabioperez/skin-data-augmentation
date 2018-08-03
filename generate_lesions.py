import argparse
import csv
import os

from itertools import combinations
from skimage.transform import resize
import random

from skimage.io import imread, imsave

from auglib.augmentation.two_lesions import mix_lesions


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = random.sample(range(n), r)
    return tuple(pool[i] for i in indices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_csv',
                        help='Path to the ground truth csv file')
    parser.add_argument('images_path',
                        help='Path to original images directory')
    parser.add_argument('n', type=int, help='Number of lesions to generate')
    parser.add_argument('--output', '-o', help='Path to output directory',
                        default='output')
    parser.add_argument('--sigma', '-s', type=float,
                        help='Gauss sigma (default: 20.0)',
                        default=20.0)
    return parser.parse_args()


def main(gt_file, dataset_path, results_path, n, sigma):
    TARGET_SIZE = (767, 1022)
    os.makedirs(results_path, exist_ok=True)
    IMAGES_PATH = os.path.join(dataset_path, 'images')
    MASKS_PATH = os.path.join(dataset_path, 'masks')

    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        dataset = [{'image_id': row['image_id'],
                    'melanoma': bool(row['melanoma'] == '1.0')}
                   for row in reader]

    for img1, img2 in random_combination(combinations(dataset, 2), n):
        print(img1['image_id'], img2['image_id'])
        img1_data = imread(os.path.join(IMAGES_PATH, img1['image_id']+'.jpg'))
        img2_data = imread(os.path.join(IMAGES_PATH, img2['image_id']+'.jpg'))
        img1_mask = imread(os.path.join(
            MASKS_PATH, img1['image_id']+'_segmentation.png'))
        img2_mask = imread(os.path.join(
            MASKS_PATH, img2['image_id']+'_segmentation.png'))
        img1_data = resize(img1_data, TARGET_SIZE,
                           preserve_range=True).astype('uint8')
        img2_data = resize(img2_data, TARGET_SIZE,
                           preserve_range=True).astype('uint8')
        img1_mask = resize(img1_mask, TARGET_SIZE,
                           preserve_range=True).astype('uint8')
        img2_mask = resize(img2_mask, TARGET_SIZE,
                           preserve_range=True).astype('uint8')

        result = mix_lesions(img1_data, img2_data, img1_mask, img2_mask,
                             gauss_sigma=sigma)
        result_name = "{}_{}_{}.jpg".format(
            img1['image_id'], img2['image_id'],
            img1['melanoma'] or img2['melanoma'])
        imsave(os.path.join(results_path, result_name), result)


if __name__ == '__main__':
    args = parse_args()
    main(args.ground_truth_csv, args.images_path, args.output, args.n,
         args.sigma)
