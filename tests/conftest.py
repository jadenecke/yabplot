import pytest
import numpy as np
import nibabel as nib
import pyvista as pv

pv.OFF_SCREEN = True

@pytest.fixture
def synthetic_nifti(tmp_path):
    """Generates a simple 3D NIfTI file with a sphere of high intensity."""
    shape = (20, 20, 20)
    data = np.zeros(shape)

    # create a simple sphere
    cz, cy, cx = 10, 10, 10
    r = 5
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= r**2
    data[mask] = 10.0

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    file_path = tmp_path / "synthetic.nii.gz"
    nib.save(img, file_path)
    return str(file_path)

@pytest.fixture
def synthetic_nifti_4d(tmp_path):
    """Generates a 4D NIfTI file."""
    shape = (20, 20, 20, 3)
    data = np.random.rand(*shape)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    file_path = tmp_path / "synthetic_4d.nii.gz"
    nib.save(img, file_path)
    return str(file_path)

@pytest.fixture
def synthetic_tractogram(tmp_path):
    """Generates a simple tractogram."""
    from nibabel.streamlines.tractogram import Tractogram
    from nibabel.streamlines.trk import TrkFile

    streamlines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
        np.array([[10, 10, 10], [11, 10, 10], [12, 10, 10]], dtype=np.float32)
    ]
    tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    file_path = tmp_path / "synthetic.trk"
    TrkFile(tractogram).save(file_path)
    return str(file_path)
