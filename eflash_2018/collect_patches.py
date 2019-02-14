"""collect-patches - given a list of points, collect the patches around them

"""

import h5py
import multiprocessing
import numpy as np
import pickle
from phathom.utils import SharedMemory
import json
import argparse
import glob
import tqdm
from .utils import RollingBuffer

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        required=True,
                        help="The glob expression for the stack")
    parser.add_argument("--points",
                        required=True,
                        help="A .json file containing the points in xyz order")
    parser.add_argument("--patch-size",
                        type=int,
                        default=31,
                        help="Size of a patch in pixels (an odd # please)")
    parser.add_argument("--output",
                        required=True,
                        help="The location for an HDF file file containing "
                        "an NxMxM array where N is the number of points and "
                        "M is the patch size and 3 arrays containing the X,"
                             "Y and Z coordinates of each of N points.")
    parser.add_argument("--n-cores",
                        default=12,
                        type=int,
                        help="The number of cores to use")
    parser.add_argument("--n-io-cores",
                        default=12,
                        type=int,
                        help="The number of cores to use during I/O")
    parser.add_argument("--model",
                        help="Model file for filtering patches.")
    parser.add_argument("--threshold",
                        default=.5,
                        type=float,
                        help="Threshold for filtering patches using model.")
    return parser.parse_args()

classifier = None
pca = None

def filter_z(pz, rb, z, half_patch_size, threshold):
    """Filter using a model

    :param pz: An array of the x, y and z center coordinates (in that order)
    :param rb: the rolling buffer
    :param z: the current Z center
    :param half_patch_size: 1/2 of the size of a patch
    :param pca: the PCA class for dimensionality reduction
    :param classifier: the random-forest classifier
    :param threshold: everything at or above this threshold passes.
    :returns: a sequence of indices that passes
    """
    all_patches = None
    for idx, (x, y) in enumerate(pz[:, :-1]):
        x, y = int(x), int(y)
        patch_xy = rb[
                       z,
                       y - half_patch_size: y + half_patch_size + 1,
                       x - half_patch_size: x + half_patch_size + 1]
        patch_xz = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y,
                           x - half_patch_size: x + half_patch_size + 1]
        patch_yz = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y - half_patch_size: y + half_patch_size + 1,
                           x]
        patches = np.hstack([patch_xy.flatten(),
                          patch_xz.flatten(),
                          patch_yz.flatten()])
        if all_patches is None:
            all_patches = np.zeros((len(pz), len(patches)), patches.dtype)
        all_patches[idx] = patches
    features = pca.transform(all_patches)
    return np.where(classifier.predict_proba(features)[:, 1] >= threshold)[0]


def do_z(pz, offset, patches_xy, patches_xz, patches_yz, rb, z,
         half_patch_size):
    for x, y in pz[:, :-1]:
        x, y = int(x), int(y)
        with patches_xy.txn() as m:
            m[offset] = rb[
                           z,
                           y - half_patch_size: y + half_patch_size + 1,
                           x - half_patch_size: x + half_patch_size + 1]
        with patches_xz.txn() as m:
            m[offset] = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y,
                           x - half_patch_size: x + half_patch_size + 1]
        with patches_yz.txn() as m:
            m[offset] = rb[
                           z - half_patch_size: z + half_patch_size + 1,
                           y - half_patch_size: y + half_patch_size + 1,
                           x]
        offset += 1
    return offset


def main():
    global classifier
    global pca
    args = parse_args()
    if args.model is not None:
        with open(args.model, "rb") as fd:
            model = pickle.load(fd)
            classifier = model["classifier"]
            pca = model["pca"]
    source_files = sorted(glob.glob(args.source))
    rb = RollingBuffer(source_files, args.n_io_cores)
    points = np.array(json.load(open(args.points)))
    patch_size = args.patch_size
    half_patch_size = patch_size // 2
    patches_xy, patches_xz, patches_yz = \
        [SharedMemory((len(points), patch_size, patch_size), rb.dtype)
         for _ in range(3)]
    points_out = []
    offset = 0
    x1 = rb.shape[2] - half_patch_size
    y1 = rb.shape[1] - half_patch_size
    with multiprocessing.Pool(args.n_cores) as pool:
        it = tqdm.tqdm(
                range(half_patch_size, len(source_files) - half_patch_size))
        for z in it:
            it.set_description("Releasing @ %d" % (z - half_patch_size))
            rb.release(z - half_patch_size)
            it.set_description("Waiting for %d" % (z + half_patch_size))
            rb.wait(z + half_patch_size)
            pz = points[(points[:, 2] >= z) & (points[:, 2] < z+1)]
            if len(pz) == 0:
                continue
            mask = np.all(pz[:, 0:2] >= half_patch_size, 1) &\
                   (pz[:, 0] < x1) &\
                   (pz[:, 1] < y1)
            pz = pz[mask]
            if len(pz) == 0:
                continue
            if len(pz) < args.n_cores * 10 or args.n_cores == 1:
                if args.model is not None:
                    idxs = filter_z(pz, rb, z, half_patch_size, args.threshold)
                    pz = pz[idxs]
                offset = do_z(pz, offset,
                              patches_xy, patches_xz, patches_yz, rb, z,
                              half_patch_size)
            else:
                idxs = np.linspace(0, len(pz), args.n_cores+1).astype(int)
                it.set_description("Freezing buffer")
                frb = rb.freeze()
                if args.model is not None:
                    it.set_description("Filtering %d patches" % len(pz))
                    fnargs = [
                        (pz[i0:i1], frb, z, half_patch_size, args.threshold)
                        for i0, i1 in zip(idxs[:-1], idxs[1:])]
                    fidxs = pool.starmap(filter_z, fnargs)
                    pzidx = np.hstack(fidxs)
                    if len(pzidx) == 0:
                        continue
                    pz = pz[pzidx]
                    if len(pz) < args.n_cores:
                        idxs = np.arange(len(pz) + 1)
                    else:
                        idxs = np.linspace(0, len(pz), args.n_cores+1).astype(int)
                cumsum = np.hstack([[0], np.cumsum([len(_) for _ in fidxs])])
                fnargs = [(pz[i0:i1], offset + i0,
                           patches_xy, patches_xz, patches_yz, frb, z,
                           half_patch_size)
                          for i0, i1 in zip(idxs[:-1], idxs[1:])]
                it.set_description("Processing %d patches @ %d" % (len(pz), z))
                pool.starmap(do_z, fnargs)
                offset += cumsum[-1]
                pz = pz[np.hstack(fidxs)]
            points_out.append(pz)

    points_out = np.vstack(points_out)
    with h5py.File(args.output, "w") as f:
        with patches_xy.txn() as m:
            old_patches = f.create_dataset("patches_xy", data=m[:offset])
        with patches_xz.txn() as m:
            f.create_dataset("patches_xz", data=m[:offset])
        with patches_yz.txn() as m:
            f.create_dataset("patches_yz", data=m[:offset])
        f.create_dataset("x", data=points_out[:, 0])
        f.create_dataset("y", data=points_out[:, 1])
        f.create_dataset("z", data=points_out[:, 2])
        f["patches"] = old_patches


if __name__ == "__main__":
    main()
