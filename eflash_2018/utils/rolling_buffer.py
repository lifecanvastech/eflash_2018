from phathom.utils import SharedMemory
import numpy as np
import tifffile
import multiprocessing


def read_plane(sm:SharedMemory, filename:str):
    """Read a plane from a TIFF file into shared memory

    :param sm: shared memory of the correct size and dtype to hold the plane
    :param filename: name of file containing the TIFF plane
    """
    with sm.txn() as memory:
        memory[:] = tifffile.imread(filename)


class RollingBuffer:
    """A rolling buffer of images

    The rolling buffer maintains a number of consecutive planes in a circular
    buffer. The buffer is filled using multiprocessing with a given number of planes
    always available and other planes being read. The user organizes their data,
    e.g. a list of patch centers, consecutively in Z, waiting for planes at higher
    z and releasing planes at lower Z.
    """

    READY = "ready"

    def __init__(self, files, n_cores):
        self.files = files
        self.n_cores = n_cores
        first_plane = tifffile.imread(files[0])
        self.x_extent = first_plane.shape[1]
        self.y_extent = first_plane.shape[0]
        self.dtype = first_plane.dtype
        self.z0 = 0
        self.z1 = 1
        self.zfill = 1
        self.planes = [ None ] * len(files)
        self.futures = [ None ] * len(files)
        self.futures[0] = self.READY
        self.free_memory = []
        self.pool = multiprocessing.Pool(n_cores)
        self.planes[0] = self.get_shared_memory()
        with self.planes[0].txn() as memory:
            memory[:] = first_plane
        self._fill(min(self.shape[0], n_cores+1))

    def get_shared_memory(self):
        if len(self.free_memory) == 0:
            return SharedMemory((self.y_extent, self.x_extent),
                                self.dtype)
        else:
            return self.free_memory.pop(0)

    def _fill(self, z_end):
        """Start the asynchronous fill operation

        :param z_end: start asynchronous acquisitions for planes up to z_end
        """
        for z in range(self.zfill, z_end):
            filename = self.files[z]
            self.planes[z] = self.get_shared_memory()
            self.futures[z] = self.pool.apply_async(
                read_plane, (self.planes[z], filename)
            )
            self.zfill = z_end

    def _wait(self, z):
        """Wait for plane Z to arrive

        :param z: The index of the plane to wait for
        """
        if z < self.z0:
            raise ValueError("Plane %d has already been released" % z)
        if z < self.z1:
            return
        if z >= self.zfill:
            self._fill(z+1)
        for zz in range(self.z1, z+1):
            if self.futures[zz] == self.READY:
                continue
            self.futures[zz].get()
            self.futures[zz] = self.READY
        self.z1 = z+1

    def release(self, z):
        """Release all memory before this Z

        :param z: The first Z needed in the rolling buffer
        """
        self.z0 = z
        while z > 0 and isinstance(self.planes[z - 1], SharedMemory):
            z = z - 1
            self.free_memory.append(self.planes[z])
            self.planes[z] = None

    @property
    def shape(self):
        return (len(self.files), self.y_extent, self.x_extent)

    def __getitem__(self, idxs):

        assert (len(idxs) == 3), "The rolling buffer is a 3D array"
        zidx = idxs[0]
        if isinstance(zidx, slice):
            start = 0 if zidx.start is None else zidx.start
            stop = \
                self.shape[0] if zidx.stop is None \
                else self.shape[0] + zidx.stop if zidx.stop < 0 \
                else zidx.stop
            step = zidx.step if zidx.step is not None else 1
            return np.array([
                self[z, idxs[1], idxs[2]]
                for z in range(start, stop, step)])
        self._wait(zidx)
        with self.planes[zidx].txn() as memory:
            return memory[idxs[1], idxs[2]]