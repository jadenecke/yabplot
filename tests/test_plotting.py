import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
import yabplot as yab
import pyvista as pv

# tell PyVista to run in "off-screen" mode so it doesn't try to open a real window
pv.OFF_SCREEN = True

def test_version():
    """Check that the package has a version string."""
    assert yab.__version__ is not None

def test_none_returns_empty_dict():
    """Verify that passing None disables the background mesh."""
    result = yab.mesh.load_bmesh(None)
    assert result == {}

def test_dict_passthrough():
    """Verify that custom dictionary keys for hemispheres are properly sanitized."""
    mesh_l = pv.Sphere()
    mesh_r = pv.Cube()
    mesh_other = pv.Cone()
    d = {'left': mesh_l, 'RIGHT': mesh_r, 'other': mesh_other}
    result = yab.mesh.load_bmesh(d)
    expected = {'L': mesh_l, 'R': mesh_r, 'other': mesh_other}
    assert result == expected

def test_polydata_wrapped_in_both():
    """Verify that passing a single PyVista mesh safely wraps it."""
    mesh = pv.Sphere()
    result = yab.mesh.load_bmesh(mesh)
    assert 'both' in result
    assert result['both'] is mesh

def test_plotter_instantiation():
    """Smoke test: can we create a Plotter without crashing?"""
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show()
    plotter.close()

def test_plot_cortical():
    """Smoke test: verify that plot_cortical renders with standard atlas."""
    yab.plot_cortical(atlas='aparc', display_type='matplotlib')

def test_plot_subcortical():
    """Smoke test: verify that plot_subcortical renders with standard atlas."""
    yab.plot_subcortical(atlas='aseg', display_type='matplotlib')

def test_plot_tracts():
    """Smoke test: verify that plot_tracts renders with standard atlas."""
    yab.plot_tracts(atlas='xtract_tiny', display_type='matplotlib')

def test_plot_vertexwise():
    """Smoke test: verify that plot_vertexwise renders with custom meshes."""
    lh = pv.Sphere()
    rh = pv.Sphere()
    lh['Data'] = np.random.rand(lh.n_points)
    rh['Data'] = np.random.rand(rh.n_points)
    yab.plot_vertexwise(lh, rh, display_type='matplotlib')

def test_plot_voxelwise(synthetic_nifti):
    """Smoke test: verify that plot_voxelwise renders with synthetic volume."""
    yab.plot_voxelwise(synthetic_nifti, threshold=5.0, blur_sigma=0, display_type='matplotlib')

def test_plot_connectome():
    """Smoke test: verify that plot_connectome renders both nodes-only and matrix modes."""
    yab.plot_connectome(atlas='aparc', display_type='matplotlib')
    # plotting with a dummy matrix
    regions = yab.get_atlas_regions('aparc', 'cortical')
    n = len(regions)
    matrix = np.random.rand(n, n)
    yab.plot_connectome(matrix=matrix, atlas='aparc', edge_threshold=0.5, display_type='matplotlib')


def test_matplotlib_ax_compatibility():
    """Verify that plotting on a provided Matplotlib axis works correctly."""
    fig, ax = plt.subplots()
    ret_ax = yab.plot_cortical(atlas='aparc', ax=ax, display_type='matplotlib')
    assert ret_ax is ax
    # verify ax is usable
    ret_ax.set_title("Test Title")
    assert ret_ax.get_title() == "Test Title"
    plt.close(fig)

def test_export_path(tmp_path):
    """Verify that export_path correctly saves a file to disk."""
    out_file = tmp_path / "test_export.png"
    yab.plot_cortical(atlas='aparc', display_type='matplotlib', export_path=str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 0
