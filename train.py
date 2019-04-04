from itertools import islice
import os

import pandas as pd
import pretrainedmodels as ptm
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from auglib.augmentation import Augmentations, set_seeds
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName
from auglib.meters import AverageMeter
from auglib.test import test_with_augmentation

ex = Experiment()
fs_observer = FileStorageObserver.create('results')
ex.observers.append(fs_observer)

TELEGRAM_KEY = 'telegram.json'
if os.path.isfile(TELEGRAM_KEY):
    telegram_obs = TelegramObserver.from_config('telegram.json')
    ex.observers.append(telegram_obs)


@ex.config
def cfg():
    train_root = None  # path to train images
    train_csv = None  # path to train CSV
    val_root = None  # path to validation images
    val_csv = None  # path to validation CSV
    test_root = None  # path to test images
    test_csv = None  # path to test CSV
    epochs = 30  # number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = None  # model: inceptionv4, densenet161, resnet152
    val_samples = 16  # number of samples per image in validation
    test_samples = 64  # number of samples per image in test
    early_stopping_patience = 8  # patience for early stopping
    images_per_epoch = None  # number of images per epoch
    limit_data = False  # limit dataset to N images
    # augmentations
    aug = {
        'hflip': False,  # Random Horizontal Flip
        'vflip': False,  # Random Vertical Flip
        'rotation': 0,  # Rotation (in degrees)
        'shear': 0,  # Shear (in degrees)
        'scale': 1.0,  # Scale (tuple (min, max))
        'color_contrast': 0,  # Color Jitter: Contrast
        'color_saturation': 0,  # Color Jitter: Saturation
        'color_brightness': 0,  # Color Jitter: Brightness
        'color_hue': 0,  # Color Jitter: Hue
        'random_crop': False,  # Random Crops
        'random_erasing': False,  # Random Erasing
        'piecewise_affine': False,  # Piecewise Affine
        'tps': False,  # TPS Affine
        'autoaugment': False # AutoAugmentation
    }


def train_epoch(device, model, dataloaders, criterion, optimizer,
                batches_per_epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    model.train()

    if batches_per_epoch:
        # Another option would be to use a PyTorch Sampler.
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders['train'])

    for data in tqdm_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    auc = roc_auc_score(all_labels, all_preds)

    return {'loss': losses.avg, 'auc': auc, 'acc': accuracies.avg}


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


@ex.automain
def main(train_root, train_csv, val_root, val_csv, test_root, test_csv,
         epochs, aug, model_name, batch_size, num_workers, val_samples,
         test_samples, early_stopping_patience, limit_data, images_per_epoch,
         _run):
    assert(model_name in ('inceptionv4', 'resnet152', 'densenet161',
                          'senet154'))

    AUGMENTED_IMAGES_DIR = os.path.join(fs_observer.dir, 'images')
    CHECKPOINTS_DIR = os.path.join(fs_observer.dir, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best.pth')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last.pth')
    RESULTS_CSV_PATH = os.path.join('results', 'results.csv')
    EXP_NAME = _run.meta_info['options']['--name']
    EXP_ID = _run._id

    for directory in (AUGMENTED_IMAGES_DIR, CHECKPOINTS_DIR):
        os.makedirs(directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)
        aug['size'] = 299
        aug['mean'] = model.mean
        aug['std'] = model.std
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
    elif model_name == 'senet154':
        model = ptm.senet154(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)
        aug['size'] = model.input_size[1]
        aug['mean'] = model.mean
        aug['std'] = model.std
    model.to(device)

    augs = Augmentations(**aug)
    model.aug_params = aug

    datasets = {
        'samples': CSVDataset(train_root, train_csv, 'image_id', 'melanoma',
                              transform=augs.tf_augment, add_extension='.jpg',
                              limit=(400, 433)),
        'train': CSVDataset(train_root, train_csv, 'image_id', 'melanoma',
                            transform=augs.tf_transform, add_extension='.jpg',
                            random_subset_size=limit_data),
        'val': CSVDatasetWithName(
            val_root, val_csv, 'image_id', 'melanoma',
            transform=augs.tf_transform, add_extension='.jpg'),
        'test': CSVDatasetWithName(
            test_root, test_csv, 'image_id', 'melanoma',
            transform=augs.tf_transform, add_extension='.jpg'),
        'test_no_aug': CSVDatasetWithName(
            test_root, test_csv, 'image_id', 'melanoma',
            transform=augs.no_augmentation, add_extension='.jpg'),
        'test_144': CSVDatasetWithName(
            test_root, test_csv, 'image_id', 'melanoma',
            transform=augs.inception_crop, add_extension='.jpg'),
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            worker_init_fn=set_seeds),
        'samples': DataLoader(datasets['samples'], batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              worker_init_fn=set_seeds),
    }

    save_images(datasets['samples'], to=AUGMENTED_IMAGES_DIR, n=32)
    sample_batch, _ = next(iter(dataloaders['samples']))
    save_image(make_grid(sample_batch, padding=0),
               os.path.join(AUGMENTED_IMAGES_DIR, 'grid.jpg'))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=0.001,
                          momentum=0.9,
                          weight_decay=0.001)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[10],
                                               gamma=0.1)
    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc']),
        'val': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
    }

    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    if images_per_epoch:
        batches_per_epoch = images_per_epoch // batch_size
    else:
        batches_per_epoch = None

    for epoch in range(epochs):
        print('train epoch {}/{}'.format(epoch+1, epochs))
        epoch_train_result = train_epoch(
            device, model, dataloaders, criterion, optimizer,
            batches_per_epoch)

        metrics['train'] = metrics['train'].append(
            {**epoch_train_result, 'epoch': epoch}, ignore_index=True)
        print('train', epoch_train_result)

        epoch_val_result, _ = test_with_augmentation(
            model, datasets['val'], device, num_workers, val_samples)

        metrics['val'] = metrics['val'].append(
            {**epoch_val_result, 'epoch': epoch}, ignore_index=True)
        print('val', epoch_val_result)
        print('-' * 40)

        scheduler.step()

        if epoch_val_result['auc'] > best_val_auc:
            best_val_auc = epoch_val_result['auc']
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > early_stopping_patience:
            torch.save(model, LAST_MODEL_PATH)
            break

        if epoch == (epochs-1):
            torch.save(model, LAST_MODEL_PATH)

    for phase in ['train', 'val']:
        metrics[phase].epoch = metrics[phase].epoch.astype(int)
        metrics[phase].to_csv(os.path.join(fs_observer.dir, phase + '.csv'),
                              index=False)

    # Run testing
    test_result, _ = test_with_augmentation(
        torch.load(BEST_MODEL_PATH), datasets['test'], device,
        num_workers, test_samples)
    print('test', test_result)

    test_noaug_result, _ = test_with_augmentation(
        torch.load(BEST_MODEL_PATH), datasets['test_no_aug'], device,
        num_workers, 1)
    print('test (no augmentation)', test_noaug_result)

    test_144crop_result, _ = test_with_augmentation(
        torch.load(BEST_MODEL_PATH), datasets['test_144'], device,
        num_workers, 1)
    print('test (144-crop)', test_144crop_result)

    with open(RESULTS_CSV_PATH, 'a') as file:
        file.write(','.join((
            EXP_NAME, str(EXP_ID), str(best_epoch), str(best_val_auc),
            str(test_noaug_result['auc']), str(test_result['auc']),
            str(test_144crop_result['auc']))) + '\n')

    return (test_noaug_result['auc'],
            test_result['auc'],
            test_144crop_result['auc'])
