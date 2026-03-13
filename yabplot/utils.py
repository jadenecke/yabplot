import os
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import scipy.sparse as sp
import matplotlib.pyplot as plt
from importlib.resources import files

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

def load_vertexwise_mesh(mesh_lh, mesh_rh, data_lh, data_rh):
    """Helper to load arbitrary user-supplied mesh (e.g., fsaverage5)."""
    lh = make_cortical_mesh(*load_gii(mesh_lh), data_lh)
    rh = make_cortical_mesh(*load_gii(mesh_rh), data_rh)
    return lh, rh

def make_cortical_mesh(verts, faces, scalars):
    """Helper to create a PyVista mesh from raw buffers."""
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten().astype(int)
    mesh = pv.PolyData(verts, faces_pv)
    mesh['Data'] = scalars
    return mesh

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

def map_values_to_surface(data, target_labels, lut_ids, dense_lut_names):
    """maps data to vertices."""
    # filter valid regions
    valid_ids_list = []
    valid_names_list = []
    
    for rid in lut_ids:
        if rid < len(dense_lut_names):
            valid_ids_list.append(rid)
            valid_names_list.append(dense_lut_names[rid])
    
    valid_ids = np.array(valid_ids_list)
    n_regions = len(valid_ids)

    # atlas visualization without data
    if data is None:
        return target_labels

    # data mapping
    max_id = max(target_labels.max(), lut_ids.max())
    lookup_table = np.full(max_id + 1, np.nan)
    source_values = np.full(n_regions, np.nan)

    if isinstance(data, dict):
        for i, name in enumerate(valid_names_list):
            if name in data:
                source_values[i] = data[name]            
    elif isinstance(data, (np.ndarray, list, tuple)):
        # map by order
        if len(data) != n_regions:
            raise ValueError(
                f"Data length mismatch! The atlas LUT defines {n_regions} regions, "
                f"but input data has {len(data)}.\n"
                f"Expected order starts with: {valid_names_list[0:3]}...\n"
                f"Solution: Use a dictionary for partial data, or check `yabplot.get_atlas_regions`."
            )
        source_values = np.array(data)
    else:
        raise ValueError("Data must be dict, list, or numpy array.")

    lookup_table[valid_ids] = source_values
    return lookup_table[target_labels]

def get_adj(faces, n_v):
    """build adjacency matrix from faces."""
    row, col = [], []
    for tri in faces:
        row.extend([tri[0], tri[1], tri[2], tri[0], tri[1], tri[2]])
        col.extend([tri[1], tri[2], tri[0], tri[2], tri[0], tri[1]])
    adj = sp.csc_matrix((np.ones_like(row), (row, col)), shape=(n_v, n_v))
    adj.data = np.ones_like(adj.data)
    return adj

def get_smooth_mask(faces, data, iterations=4):
    """blur binary mask for guide of geometric slicing."""
    n_v = len(data)
    mask = data.astype(np.float64)
    adj = get_adj(faces, n_v)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0 
    for _ in range(iterations):
        mask = (mask + (adj.dot(mask) / deg)) / 2.0
    return mask

def apply_internal_blur(faces, data, iterations=1, weight=0.2):
    """blur data only on borders where different regions touch."""
    data_out = np.copy(data)
    n_v = len(data)
    adj = get_adj(faces, n_v)
    rows, cols = adj.nonzero()
    valid = ~np.isnan(data_out)
    diff = valid[rows] & valid[cols] & ~np.isclose(data_out[rows], data_out[cols], atol=1e-5)
    b_verts = np.unique(rows[diff])
    
    if len(b_verts) == 0: return data_out

    for _ in range(iterations):
        temp = np.nan_to_num(data_out, nan=0.0)
        v_counts = adj.dot(valid.astype(float))
        v_counts[v_counts == 0] = 1.0
        n_mean = adj.dot(temp) / v_counts
        data_out[b_verts] = (1 - weight) * data_out[b_verts] + weight * n_mean[b_verts]
    return data_out

def apply_dilation(faces, data, iterations=4):
    """push values into NaN space to keep geometric cut pure."""
    data_out = np.copy(data)
    n_v = len(data)
    adj = get_adj(faces, n_v)
    for _ in range(iterations):
        nan_m = np.isnan(data_out)
        temp = np.nan_to_num(data_out, nan=0.0)
        v_counts = adj.dot((~nan_m).astype(float))
        s_neighbors = adj.dot(temp)
        u_mask = nan_m & (v_counts > 0)
        data_out[u_mask] = s_neighbors[u_mask] / v_counts[u_mask]
    return data_out

def get_puzzle_pieces(v, f, raw_vals):
    """carve out geometric pieces with slight overlap to prevent gaps."""
    pieces = []
    valid_mask = ~np.isnan(raw_vals) & (raw_vals != 0.0)
    u_vals = np.unique(raw_vals[valid_mask])
    master = make_cortical_mesh(v, f, np.zeros_like(raw_vals))

    for val in u_vals:
        r_mask = np.where(raw_vals == val, 1.0, 0.0)
        s_mask = get_smooth_mask(f, r_mask, iterations=4)
        temp = master.copy()
        temp['Slice_Mask'] = s_mask
        # reduce search space
        patch = temp.threshold(0.01, scalars='Slice_Mask')
        if patch.n_points > 0:
            # use 0.48 (slightly expanded) for pieces to seal cracks
            piece = patch.clip_scalar(scalars='Slice_Mask', value=0.48, invert=False)
            if piece.n_points > 0:
                piece['Data'] = np.full(piece.n_points, val)
                pieces.append(piece)
    
    # slice base brain
    all_mask = np.where(valid_mask, 1.0, 0.0)
    s_all = get_smooth_mask(f, all_mask, iterations=4)
    master['Slice_Mask'] = s_all
    # use 0.52 (slightly contracted) for the hole to ensure colored pieces cover the edge
    base_p = master.clip_scalar(scalars='Slice_Mask', value=0.52, invert=True)
    if base_p.n_points > 0:
        base_p['Data'] = np.full(base_p.n_points, np.nan)
    
    return base_p, pieces

def lines_from_streamlines(streamlines):
    if len(streamlines) == 0: return np.array([]), np.array([]), np.array([])
    
    points = np.vstack(streamlines)
    n_points = [len(s) for s in streamlines]
    offsets = np.insert(np.cumsum(n_points), 0, 0)[:-1]
    
    cells = []
    for length, offset in zip(n_points, offsets):
        cells.append(np.hstack([[length], np.arange(offset, offset + length)]))
    lines = np.hstack(cells)
    
    # calculate tangents
    tangents = []
    for s in streamlines:
        if len(s) < 2: 
            tangents.append(np.array([[0,0,0]]))
            continue
        vecs = np.diff(s, axis=0)
        vecs = np.vstack([vecs, vecs[-1:]])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tangents.append(vecs / norms)
        
    return points, lines, np.vstack(tangents)