import contextlib
import numpy as np
import tempfile
import tifffile
import unittest
from eflash_2018.utils import RollingBuffer

@contextlib.contextmanager
def make_case(shape):
    """Make a test case

    :param shape: the shape of the volume to test, (z, y, x)
    """
    z, y, x = shape
    volume = np.random.randint(0, 65535, shape).astype(np.uint16)
    tempfiles = [tempfile.NamedTemporaryFile(suffix=".tiff")
                 for _ in range(z)]
    for plane, tf in zip(volume, tempfiles):
        tifffile.imsave(tf.name, plane)
    yield RollingBuffer([_.name for _ in tempfiles], 1), volume
    for tf in tempfiles:
        tf.close()



class TestRollingBuffer(unittest.TestCase):

    def test_shape(self):
        with make_case((5, 6, 7)) as (rb, volume):
            self.assertSequenceEqual(rb.shape, (5, 6, 7))

    def test_first_z(self):
        with make_case((1, 100, 100)) as (rb, volume):
            np.testing.assert_array_equal(rb[0, :, :], volume[0])

    def test_other_z(self):
        with make_case((2, 100, 100)) as (rb, volume):
            np.testing.assert_array_equal(rb[1, :, :], volume[1])

    def test_from_z_start(self):
        with make_case((3, 100, 100)) as (rb, volume):
            np.testing.assert_array_equal(rb[:2, :, :], volume[:2])

    def test_to_z_end(self):
        with make_case((3, 100, 100)) as (rb, volume):
            np.testing.assert_array_equal(rb[1:, :, :], volume[1:])

    def test_negative_z_end(self):
        with make_case((3, 100, 100)) as (rb, volume):
            np.testing.assert_array_equal(rb[:-1, :, :], volume[:-1])

    def test_x_plane(self):
        with make_case((10, 10, 10)) as (rb, volume):
            np.testing.assert_array_equal(rb[:, :, 5], volume[:, :, 5])

    def test_y_plane(self):
        with make_case((10, 10, 10)) as (rb, volume):
            np.testing.assert_array_equal(rb[:, 5, :], volume[:, 5, :])

    def test_rolling_buffer(self):
        with make_case((10, 10, 10)) as (rb, volume):
            np.testing.assert_array_equal(rb[:5, :, :], volume[:5])
            rb.release(5)
            self.assertRaises(ValueError,
                              lambda :rb[4, :, :])


class TestFrozenRollingBuffer(unittest.TestCase):
    def test_freeze(self):
        with make_case((10, 10, 10)) as (rb, volume):
            rb.wait(9)
            frb = rb.freeze()
            np.testing.assert_array_equal(frb[:, :, :], volume)\


if __name__ == '__main__':
    unittest.main()
