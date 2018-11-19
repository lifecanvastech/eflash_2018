import argparse
import glob
import json
import numpy as np
from phathom.segmentation.segmentation import find_centroids
from scipy.ndimage import label
from skimage.transform import integral_image, resize
from skimage.feature import haar_like_feature, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import tqdm

import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

base_model = inception_v3(pretrained=True).cuda()
base_model.eval()

#
# Monkey patch away the top layer
#
base_model.forward = lambda x: forward(base_model, x)

def forward(self, x):
    if self.transform_input:
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    # 299 x 299 x 3
    x = self.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.Mixed_6e(x)
    # 17 x 17 x 768
    if self.training and self.aux_logits:
        aux = self.AuxLogits(x)
    # 17 x 17 x 768
    x = self.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.Mixed_7c(x)
    # 8 x 8 x 2048
    return x

#### The following is from keras_applications.imagenet_utils
#    It's clearly from somewhere else because it handles PyTorch.
#

def _preprocess_numpy_input(x, data_format, mode, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed Numpy array.
    """
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x



def preprocess_input(x, data_format=None, mode='torch', **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images.
    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed tensor or Numpy array.
    # Raises
        ValueError: In case of unknown `data_format` argument.
    """

    data_format = 'channels_first'

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format,
                                       mode=mode, **kwargs)
    else:
        raise NotImplementedError("Only Numpy arrays supported")

# End keras_applications code
#
#######################################################


def extract_patch(img, y, x, width):
    y, x = int(round(y)), int(round(x))
    start = [y - width // 2, x - width // 2]
    stop = [y + width // 2 + 1, x + width // 2 + 1]
    return img[start[0]:stop[0], start[1]:stop[1]]


def filter_centroids(img, centroids, width, threshold):
    idxs = []
    for i, (y, x) in enumerate(centroids):
        if y < width // 2 or y >= img.shape[0] - width // 2 - 1:
            continue
        if x < width // 2 or x >= img.shape[1] - width // 2 - 1:
            continue
        if np.mean(extract_patch(img, y, x, width)) >= threshold:
            idxs.append(i)
    return np.array(idxs)


def preprocess_patch(img, max_val=None):
    if max_val is None:
        max_val = img.max()
    img = (img / max_val * 255).astype(np.uint8)
    img = resize(img, (299, 299), preserve_range=True)
    img_rgb = np.array([img, img, img])
    return preprocess_input(img_rgb)


def make_features(output):
    aves = output.mean(axis=(1, 2))
    stds = output.std(axis=(1, 2))
    features = np.concatenate((aves, stds), axis=-1)
    return features


def patch_features(img, centroids, width, model, batch_size):
    max_val = img.max()
    nb_patches = centroids.shape[0]
    features = []
    for idx in tqdm.tqdm(
            range(0, nb_patches, batch_size),
            desc="Running inception"):
        patches = []
        for y, x in centroids[idx:idx+batch_size]:
            patch = extract_patch(img, y, x, width)
            x = preprocess_patch(patch, max_val)
            patches.append(x)
        patch_batch = np.array(patches)
        with torch.no_grad():
            outputs = model(torch.Tensor(patch_batch).cuda()).cpu().numpy()
            for output in outputs:
                features.append(make_features(output))
    return np.vstack(features)


def patch_mfi(img, centroids, width):
    nb_patches = centroids.shape[0]
    features = np.zeros((nb_patches, 2))
    for i, (y, x) in enumerate(centroids):
        patch = extract_patch(img, y, x, width)
        features[i] = np.array([patch.mean(), patch.std()])
    return features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syto",
                        required=True,
                        help="Glob expression for syto files")
    parser.add_argument("--antibody",
                        required=True,
                        help="Glob expression for antibody stain channel files")
    parser.add_argument("--output",
                        required=True,
                        help="Path to output file")
    parser.add_argument("--hessian-sigma",
                        type=float,
                        default=4.0,
                        help="Sigma for hessian matrix")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="# of patches per inception run")
    parser.add_argument("--min-antibody-intensity",
                        type=float,
                        default=100,
                        help="Minimum mean intensity of antibody channel "
                        "in vicinity of center.")
    parser.add_argument("--patch-width",
                        type=int,
                        default=31,
                        help="Size of a patch")
    parser.add_argument("--sample-size",
                        type=int,
                        default=1000000,
                        help="Subsample this many centers for PCA")
    return parser.parse_args()


def main():
    args = parse_args()
    width = args.patch_width
    syto_files = sorted(glob.glob(args.syto))
    antibody_files = sorted(glob.glob(args.antibody))
    assert len(syto_files) > 0, "No files found at %s" % args.syto
    assert len(antibody_files) > 0, "No files found at %s" % args.antibody
    assert len(syto_files) == len(antibody_files),\
           "%d syto files, %d antibody files. Number differs!" %\
           (len(syto_files), len(antibody_files))

    all_centers = []
    all_features = []
    all_pixels = []
    for z, (syto_file, antibody_file) in \
            tqdm.tqdm(enumerate(zip(syto_files, antibody_files)),
                      desc="Z stack",
                      total = len(syto_files)):
        syto_slice = tifffile.imread(syto_file)
        antibody_slice = tifffile.imread(antibody_file)
        hessian = hessian_matrix(syto_slice, args.hessian_sigma)
        eigvals = hessian_matrix_eigvals(hessian)
        eigvals = np.clip(eigvals, None, 0)
        threshold = -threshold_otsu(-eigvals[0])
        mask = (eigvals[0] < threshold)
        lbl, nb_lbls = label(mask)
        centroids = find_centroids(lbl[np.newaxis])[:, 1:3]
        idxs = filter_centroids(antibody_slice,
                                centroids,
                                width,
                                args.min_antibody_intensity)
        if len(idxs) == 0:
            continue
        centroids = centroids[idxs]
        antibody_features = patch_features(antibody_slice,
                                           centroids,
                                           width,
                                           base_model,
                                           args.batch_size)
        all_centers.append(np.column_stack((
            np.ones(len(centroids)) * z, centroids[:, 0], centroids[:, 1])))
        all_features.append(antibody_features)
    all_centers = np.concatenate(all_centers)
    all_features = np.concatenate(all_features)

    n_samples = len(all_features)
    if n_samples > args.sample_size:
        idxs = np.random.choice(n_samples, args.sample_size)
        sample = all_features[idxs]
    else:
        sample = all_features
    pca = PCA(n_components = 8)
    pca.fit(sample)
    pca_features = pca.transform(all_features)
    gmm = GaussianMixture(n_components=2).fit(pca_features)
    labels = gmm.predict(pca_features)
    with open(args.output, "w") as fd:
        json.dump(dict(x=all_centers[:, 2].tolist(),
                       y=all_centers[:, 1].tolist(),
                       z=all_centers[:, 0].tolist(),
                       labels=labels.tolist(),
                       pca_features=pca_features.tolist()), fd)


if __name__ == "__main__":
    main()