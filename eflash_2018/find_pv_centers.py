import argparse
import base64
import glob
import json
import numpy as np
from phathom.segmentation.segmentation import find_centroids
from phathom.preprocess.filtering import clahe_2d
import pickle
from scipy.ndimage import label
from skimage.transform import integral_image, resize
from skimage.feature import haar_like_feature, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import tqdm
import tifffile

import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

base_model = InceptionV3(weights="imagenet", include_top=False)


def extract_patch(img, y, x, width):
    y, x = int(round(y)), int(round(x))
    start = [y - width // 2, x - width // 2]
    stop = [y + width // 2 + 1, x + width // 2 + 1]
    return img[start[0]:stop[0], start[1]:stop[1]]


def filter_centroids(img, centroids, width, patch_width, threshold, frac):
    """Filter centroids that are near edge or low antibody intensity

    :param img: antibody image
    :param centroids: y, x coordinates of cells to filter
    :param width: Width of inner patch
    :param patch_width: width of outer patch / patch to be taken
    :param threshold: Accept cells only if mean antibody intensity is greater
    than this
    :param frac: If the outside mean intensity times the fraction is greater
    than the inside mean intensity, don't accept the cell.
    :return: the indexes of the centroids to take.
    """
    idxs = []
    for i, (y, x) in enumerate(centroids):
        if y < patch_width // 2 or y >= img.shape[0] - patch_width // 2 - 1:
            continue
        if x < patch_width // 2 or x >= img.shape[1] - patch_width // 2 - 1:
            continue
        inner_mean = np.mean(extract_patch(img, y, x, width))
        outer_mean = np.mean(extract_patch(img, y, x, patch_width))
        if outer_mean * frac > inner_mean:
            continue
        if inner_mean >= threshold:
            idxs.append(i)
    return np.array(idxs)


def preprocess_patch(img, max_val=None):
    if max_val is None:
        max_val = img.max()
    img = (img / max_val * 255).astype(np.uint8)
    img = resize(img, (299, 299), preserve_range=True)
    img_rgb = np.dstack([img, img, img])
    return preprocess_input(img_rgb)


dgrid = np.sqrt(np.square(np.linspace(-1, 1, 8).reshape(8, 1)) +
                np.square(np.linspace(-1, 1, 8).reshape(1, 8)))
dgrid = (dgrid / np.sum(dgrid)).reshape(8, 8, 1)


def make_features(output):
    aves = output.mean(axis=(0, 1))
    stds = output.std(axis=(0, 1))
    skews = np.sum(output * dgrid, (0, 1))
    features = np.concatenate((aves, skews, stds), axis=-1)
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
        outputs = model.predict(patch_batch)
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
    parser.add_argument("--centers",
                        help="A .json file containing a list of lists of "
                        "cell centers organized as x, y, z. If present, "
                        "this list is used instead of the 2D curvature.")
    parser.add_argument("--model-path",
                        help="Path to a prior run's output file. Use the "
                        "PCA and GMM model contained in the file.")
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
                        default=.15,
                        help="Minimum mean intensity of antibody channel "
                        "in vicinity of center.")
    parser.add_argument("--patch-width",
                        type=int,
                        default=31,
                        help="Size of a patch")
    parser.add_argument("--intensity-patch-width",
                        type=int,
                        default=15,
                        help="Size of patch for antibody intensity calculation")
    parser.add_argument("--min-intensity-frac",
                        type=float,
                        default=1.0,
                        help="Cells are rejected if the antibody intensity "
                             "in the middle patch is less than this fraction of"
                             " the outside intensity")
    parser.add_argument("--sample-size",
                        type=int,
                        default=1000000,
                        help="Subsample this many centers for PCA")
    parser.add_argument("--clahe-kernel-size",
                        type=int,
                        default=127,
                        help="Shape of the contextual regions for clahe "
                        "intensity normalization")
    parser.add_argument("--clahe-clip-limit",
                        type=float,
                        default=.01,
                        help="Fractional number of pixels to clip for clahe "
                        "intensity normalization")
    parser.add_argument("--clahe-num-bins",
                        type=int,
                        default=256,
                        help="Number of bins for clahe intensity normalization")
    parser.add_argument("--n-pca-components",
                        type=int,
                        default=8,
                        help="Number of PCA components during dimensionality "
                        "reduction.")
    parser.add_argument("--n-gmm-components",
                        type=int,
                        default=2,
                        help="Number of gaussian mixture model components.")
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
    if args.centers is not None:
        precomputed_centers = True
        centers = np.array(json.load(open(args.centers)))[:, ::-1].astype(int)
        all_centers = []
        for z in range(len(syto_files)):
            all_centers.append(centers[centers[:, 0] == z])
    else:
        all_centers = []
        precomputed_centers = False
    if args.model_path is not None:
        old_output = json.load(open(args.model_path))
        pca = pickle.loads(
            base64.b64decode(old_output["pca_pickle"].encode("ascii")))
        gmm = pickle.loads(
            base64.b64decode(old_output["gmm_pickle"].encode("ascii")))
    all_features = []
    for z, (syto_file, antibody_file) in \
            tqdm.tqdm(enumerate(zip(syto_files, antibody_files)),
                      desc="Z stack",
                      total = len(syto_files)):
        syto_slice = tifffile.imread(syto_file)
        antibody_slice = tifffile.imread(antibody_file)
        antibody_slice = clahe_2d(antibody_slice,
                                  kernel_size=args.clahe_kernel_size,
                                  clip_limit=args.clahe_clip_limit,
                                  nbins=args.clahe_num_bins)
        if not precomputed_centers:
            hessian = hessian_matrix(syto_slice, args.hessian_sigma)
            eigvals = hessian_matrix_eigvals(hessian)
            eigvals = np.clip(eigvals, None, 0)
            threshold = -threshold_otsu(-eigvals[0])
            mask = (eigvals[0] < threshold)
            lbl, nb_lbls = label(mask)
            centroids = find_centroids(lbl[np.newaxis])[:, 1:3]
        else:
            centroids = all_centers[z][:, 1:3]
            if len(centroids) == 0:
                continue
        idxs = filter_centroids(antibody_slice,
                                centroids,
                                args.intensity_patch_width,
                                args.patch_width,
                                args.min_antibody_intensity,
                                args.min_intensity_frac)
        if len(idxs) == 0:
            continue
        centroids = centroids[idxs]
        antibody_features = patch_features(antibody_slice,
                                           centroids,
                                           width,
                                           base_model,
                                           args.batch_size)
        if not precomputed_centers:
            all_centers.append(np.column_stack((
                np.ones(len(centroids)) * z, centroids[:, 0], centroids[:, 1])))
        else:
            all_centers[z] = all_centers[z][idxs]
        all_features.append(antibody_features)
        np.save("/media/share2/Lee/2018-11-02/features/features-%04d.npy" % z,
                antibody_features)
    all_centers = np.concatenate(all_centers)
    all_features = np.concatenate(all_features)

    n_samples = len(all_features)
    if n_samples > args.sample_size:
        idxs = np.random.choice(n_samples, args.sample_size)
        sample = all_features[idxs]
    else:
        sample = all_features
    if args.model_path is None:
        pca = PCA(n_components = args.n_pca_components)
        pca.fit(sample)
    pca_features = pca.transform(all_features)
    if args.model_path is None:
        gmm = GaussianMixture(
            n_components=args.n_gmm_components).fit(pca_features)
    labels = gmm.predict(pca_features)
    pca_pickle = base64.b64encode(pickle.dumps(pca)).decode("ascii")
    gmm_pickle = base64.b64encode(pickle.dumps(gmm)).decode("ascii")
    with open(args.output, "w") as fd:
        json.dump(dict(x=all_centers[:, 2].tolist(),
                       y=all_centers[:, 1].tolist(),
                       z=all_centers[:, 0].tolist(),
                       labels=labels.tolist(),
                       pca_features=pca_features.tolist(),
                       pca_pickle=pca_pickle,
                       gmm_pickle=gmm_pickle), fd)
    print("Explained variance: " + str(pca.explained_variance_ratio_))


if __name__ == "__main__":
    main()