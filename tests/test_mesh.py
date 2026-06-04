import pytest
import numpy as np
import pyvista as pv
from yabplot.mesh import load_nii_as_mesh, make_cortical_mesh

def test_load_nii_as_mesh(synthetic_nifti):
    """Verify that marching cubes mesh extraction works on a NIfTI volume."""
    mesh = load_nii_as_mesh(synthetic_nifti, threshold=5.0, blur_sigma=0, smooth_i=0)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0
    assert mesh.n_cells > 0

def test_make_cortical_mesh():
    """Verify that a manual mesh can be constructed from vertex and face arrays."""
    verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
    faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int64)
    scalars = np.array([1, 2, 3, 4], dtype=np.float32)
    mesh = make_cortical_mesh(verts, faces, scalars, scalar_name='TestData')

    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 4
    assert mesh.n_cells == 4
    assert 'TestData' in mesh.point_data
    assert np.all(mesh.point_data['TestData'] == scalars)
