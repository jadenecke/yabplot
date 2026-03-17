import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt

def load_gii(gii_path):
    """Load GIfTI geometry (vertices, faces)."""
    mesh = nib.load(gii_path)
    verts = mesh.darrays[0].data
    faces = mesh.darrays[1].data
    return verts, faces

def load_gii2pv(gii_path, smooth_i=0, smooth_f=0.1):
    """
    Load GIfTI and convert to PyVista format with optional smoothing.
    
    Parameters
    ----------
    smooth_i : int
        Number of smoothing iterations (e.g. 15).
    smooth_f : float
        Relaxation factor (0.0 to 1.0, e.g. 0.6).
    """
    verts, faces = load_gii(gii_path)
    
    # create pyvista mesh
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten().astype(int)
    mesh = pv.PolyData(verts, faces_pv)
    
    # apply smoothing
    if smooth_i > 0:
        # use Laplacian smoothing (standard vtkSmoothPolyDataFilter)
        # note: higher relaxation factors can shrink the mesh significantly
        # if shrinkage is an issue, could consider mesh.smooth_taubin() instead
        mesh = mesh.smooth(n_iter=smooth_i, relaxation_factor=smooth_f)
    
    return mesh

def array_to_gifti(arr, out_path):
    """Save a 1D numpy array as a GIFTI metric file.

    Parameters
    ----------
    arr : np.ndarray
        1D array of shape (n_vertices,).
    out_path : str
        Output path, e.g. 'input.L.func.gii'.
    """
    darray = nib.gifti.GiftiDataArray(arr.astype(np.float32))
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, out_path)

def prep_data(data, regions, atlas, category):
    """Standardize input data to dictionary."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] >= 2:
            return dict(zip(data.iloc[:, 0], data.iloc[:, 1]))
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, dict):
        return data
    elif isinstance(data, (list, np.ndarray, tuple)):
        if len(data) != len(regions):
            raise ValueError(
                f"Data length mismatch! Atlas '{atlas}' has {len(regions)} regions, "
                f"but input data has {len(data)}. "
                f"For partial data, use a dictionary, pd.Series, or pd.DataFrame. "
                f"Use `yabplot.get_atlas_regions('{atlas}', '{category}')` to see expected order."
            )
        # map strictly by order
        return dict(zip(regions, data))

    return data

def generate_distinct_colors(n_colors, seed=42):
    """Generate visually distinct colors using Golden Ratio."""
    np.random.seed(seed)
    colors = []
    hue = np.random.rand()
    for _ in range(n_colors):
        hue = (hue + 0.618033988749895) % 1.0
        colors.append(plt.cm.hsv(hue)[:3])
    return colors

def parse_lut(lut_path):
    """parses LUT to color array and name list."""

    # load and sort by ID to ensure strict order (1..N)
    df = pd.read_csv(lut_path, sep=r'\s+', header=None)
    df = df.sort_values(by=0)
    
    ids = df[0].values
    names = df[1].tolist()
    rgb = df.iloc[:, 2:5].values / 255.0
    
    max_id = ids.max()
    
    lut_colors = np.full((max_id + 1, 3), 0.5) 
    lut_names_list = ["Unknown"] * (max_id + 1)
    
    lut_colors[ids] = rgb
    for idx, name in zip(ids, names):
        lut_names_list[idx] = name
        
    return ids, lut_colors, lut_names_list, max_id

