import pytest
import numpy as np
from yabplot.projection import project_vol2surf, project_vol2tract

def test_project_vol2surf(synthetic_nifti):
    """Verify that a 3D NIfTI can be projected onto the cortical surface."""
    lh_data, rh_data = project_vol2surf(synthetic_nifti, bmesh='midthickness', mask_medial_wall=False, interpolation='linear')
    assert isinstance(lh_data, np.ndarray)
    assert isinstance(rh_data, np.ndarray)
    assert len(lh_data) > 0
    assert len(rh_data) > 0

def test_project_vol2tract(synthetic_tractogram, synthetic_nifti):
    """Verify that a 3D NIfTI can be projected onto a tractogram."""
    data = project_vol2tract(synthetic_tractogram, synthetic_nifti, interpolation='linear')
    assert isinstance(data, np.ndarray)
    assert len(data) > 0

def test_project_invalid_interpolation(synthetic_nifti):
    """Verify that invalid interpolation modes raise an error."""
    with pytest.raises(ValueError, match="interpolation must be"):
        project_vol2surf(synthetic_nifti, bmesh='midthickness', interpolation='invalid')

    with pytest.raises(ValueError, match="interpolation must be"):
        project_vol2tract("dummy.trk", synthetic_nifti, interpolation='invalid')
