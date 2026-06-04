"""
Data management module for fetching and caching remote atlases.
"""

import os
import glob
from pathlib import Path
import pooch
import shutil

from ..utils import parse_lut

__all__ = ['get_available_resources', 'get_atlas_regions']

# define cache location
CACHE_DIR = pooch.os_cache("yabplot")

# setup registry
HERE = Path(__file__).parent
REGISTRY_FILE = HERE / "registry.txt"

GOODBOY = pooch.create(
    path=CACHE_DIR,
    base_url="",
    registry=None,
)
GOODBOY.load_registry(REGISTRY_FILE)


def get_available_resources(category=None):
    """
    Returns available resources from the registry.

    Parameters
    ----------
    category : str or None
        If provided (e.g., 'cortical', 'subcortical', 'tracts', 'bmesh'), returns a list of available names 
        for that specific category.
        If None, returns a dictionary containing all categories and their options.
    """
    if not GOODBOY.registry:
        return [] if category else {}

    # helper to clean names: e.g., "cortical-aparc.zip" -> ("cortical", "aparc")
    def _parse_key(key):
        if "-" not in key: return None, None
        prefix, remainder = key.split("-", 1)
        name = remainder.replace(".zip", "")
        return prefix, name

    # mode 1: specific category
    if category:
        available = []
        for key in GOODBOY.registry.keys():
            prefix, name = _parse_key(key)
            if prefix == category:
                available.append(name)
        return sorted(available)

    # mode 2: all categories
    all_resources = {}
    for key in GOODBOY.registry.keys():
        prefix, name = _parse_key(key)
        if prefix and name:
            if prefix not in all_resources:
                all_resources[prefix] = []
            all_resources[prefix].append(name)
    
    for k in all_resources:
        all_resources[k].sort()
        
    return all_resources

def get_surface_paths(name, category):
    """
    Fetches and returns the paths to the Left and Right hemisphere files 
    for a given surface resource (meshes or labels).
    
    Parameters
    ----------
    name : str
        Name of the resource (e.g., 'midthickness', 'nomedialwall').
    category : str
        Must be 'bmesh' or 'label'.
        
    Returns
    -------
    tuple
        (lh_path, rh_path) containing absolute paths to the files.
    """
    if category not in ['bmesh', 'label']:
        raise ValueError("Category must be 'bmesh' or 'label' to fetch surface paths.")
        
    # Download/unpack the zip and get the folder path
    directory = _resolve_resource_path(name, category)
    
    lh_path = None
    rh_path = None
    
    # Traverse the unzipped directory to find L and R files
    for root, dirs, files in os.walk(directory):
        # Ignore hidden folders like .git or __MACOSX
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        for file in files:
            # Ignore hidden files
            if file.startswith('.'): 
                continue
            
            # Robust checking for Left and Right hemisphere indicators
            if '.L.' in file or '_L_' in file or 'hemi-L' in file:
                lh_path = os.path.join(root, file)
            elif '.R.' in file or '_R_' in file or 'hemi-R' in file:
                rh_path = os.path.join(root, file)
                
    if not lh_path or not rh_path:
        raise FileNotFoundError(f"Could not locate both Left and Right hemisphere files for '{name}' in {directory}")
        
    return lh_path, rh_path

def get_atlas_regions(atlas, category, custom_atlas_path=None):
    """
    Returns the list of region names for a given atlas in the specific order 
    used for mapping data arrays.

    Parameters
    ----------
    atlas : str
        Name of the atlas (e.g., 'aparc', 'aseg').
    category : str
        'cortical', 'subcortical', or 'tracts'.
    custom_atlas_path : str, optional
        Path to custom atlas directory.

    Returns
    -------
    list
        List of strings containing region names. 
        - If input data is a LIST, it must match this order.
        - If input data is a DICT, keys must match these names.
    """
    
    # resolve the directory path
    try:
        atlas_dir = _resolve_resource_path(atlas, category, custom_path=custom_atlas_path)
    except Exception as e:
        print(f"Error resolving atlas: {e}")
        return []

    # --- case 1: cortical ---
    if category == 'cortical':
        check_name = None if custom_atlas_path else atlas
        try:
            _, lut_path = _find_cortical_files(atlas_dir, strict_name=check_name)
            
            # use parse_lut to get the IDs and the full names list
            ids, _, names_list, _ = parse_lut(lut_path)
            
            # return only the names corresponding to the explicit IDs in the file.
            return [names_list[i] for i in ids]
            
        except Exception as e:
            print(f"Error parsing cortical atlas: {e}")
            return []

    # --- case 2: subcortical ---
    elif category == 'subcortical':
        try:
            file_map = _find_subcortical_files(atlas_dir)
            return _get_ordered_names(atlas_dir, file_map)
        except Exception as e:
            print(f"Error listing subcortical regions: {e}")
            return []

    # --- case 3: tracts ---
    elif category == 'tracts':
        try:
            file_map = _find_tract_files(atlas_dir)
            return _get_ordered_names(atlas_dir, file_map)
        except Exception as e:
            print(f"Error listing tracts: {e}")
            return []

    else:
        raise ValueError("Category must be 'cortical', 'subcortical', or 'tracts'")


def _fetch_and_unpack(resource_key):
    """
    Downloads zip, unpacks it, deletes the zip to save space, 
    and returns the extraction path. Forces a redownload if the 
    registry hash changes (indicating an update).
    """
    extract_dir_name = resource_key.replace(".zip", "")
    extract_path = os.path.join(GOODBOY.path, extract_dir_name)
    hash_file = os.path.join(extract_path, ".registry_hash")

    # get the expected hash from the registry
    expected_hash = GOODBOY.registry.get(resource_key)
    if not expected_hash:
        raise ValueError(f"Resource '{resource_key}' not found in registry.")

    # check if unpacked folder already exists and is up-to-date
    is_up_to_date = False
    if os.path.isdir(extract_path) and os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            local_hash = f.read().strip()
        if local_hash == expected_hash:
            is_up_to_date = True
    if is_up_to_date:
        return extract_path
    # if folder exists but hash is wrong (outdated), wipe it clean
    elif os.path.exists(extract_path):
        print(f"Update found for '{extract_dir_name}'. Removing legacy data...")
        shutil.rmtree(extract_path)
        
    # fetch and unzip new data
    try:
        GOODBOY.fetch(
            resource_key, 
            processor=pooch.Unzip(extract_dir=extract_dir_name)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch '{resource_key}': {e}")
    
    # stamp the new folder with the updated hash
    with open(hash_file, 'w') as f:
        f.write(expected_hash)

    # cleanup: delete the source zip to save space
    zip_path = os.path.join(GOODBOY.path, resource_key)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return extract_path


def _resolve_resource_path(name, category, custom_path=None):
    """
    Internal: Resolves atlas path via download or custom location.
    """
    # 1. custom path logic
    if custom_path:
        if os.path.isdir(custom_path):
            return custom_path
        if os.path.isfile(custom_path):
            return custom_path
        raise FileNotFoundError(f"Custom atlas directory/file not found: {custom_path}")

    # 2. standard download logic
    resource_key = f"{category}-{name}.zip"
    
    # validate before fetching
    if resource_key not in GOODBOY.registry:
        available = get_available_resources(category)
        human_cat = {
            'cortical': 'Cortical parcellations (vertices)',
            'subcortical': 'Subcortical segmentations (volumes)', 
            'tracts': 'White matter bundles (tracts)',
            'bmesh': 'Brain meshes',
            'label': 'Surface labels'
        }.get(category, category)
        
        raise ValueError(
            f"Resource '{name}' is not available in {human_cat}.\n"
            f"Available options: {available}"
        )

    return _fetch_and_unpack(resource_key)


def _find_cortical_files(atlas_dir, strict_name=None):
    """
    Internal: Locates files, ignoring hidden/system folders.
    """
    
    def _find_file(directory, pattern):
        """searches root and valid subdirectories."""
        # check root
        candidates = glob.glob(os.path.join(directory, pattern))
        
        # check subdirs if empty
        if not candidates:
            try:
                # get all items, filtering out hidden/system ones
                subdirs = [
                    os.path.join(directory, d) for d in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, d)) 
                    and not d.startswith(('.', '__'))
                ]
                subdirs.sort()
                
                for sd in subdirs:
                    candidates.extend(glob.glob(os.path.join(sd, pattern)))
            except FileNotFoundError:
                pass
        
        return candidates

    # --- mode a: strict (standard atlases) ---
    if strict_name:
        csv_name = f'{strict_name}_conte69.csv'
        lut_name = f'{strict_name}_LUT.txt'
        
        found_csvs = _find_file(atlas_dir, csv_name)
        if not found_csvs:
            raise FileNotFoundError(f"Corrupt atlas. Missing '{csv_name}' in {atlas_dir}")
        
        found_luts = _find_file(atlas_dir, lut_name)
        if not found_luts:
             raise FileNotFoundError(f"Corrupt atlas. Missing '{lut_name}' in {atlas_dir}")
            
        return found_csvs[0], found_luts[0]

    # --- mode b: flexible (custom atlases) ---
    
    # find csv
    csv_candidates = _find_file(atlas_dir, "*.csv")
    if len(csv_candidates) == 1:
        csv_path = csv_candidates[0]
    elif len(csv_candidates) > 1:
        # resolve ambiguity
        filtered = [f for f in csv_candidates if 'conte69' in f]
        if len(filtered) == 1:
            csv_path = filtered[0]
        else:
            names = [os.path.basename(c) for c in csv_candidates]
            raise ValueError(f"Ambiguous CSVs found: {names}")
    else:
        raise FileNotFoundError(f"No .csv file found in custom directory: {atlas_dir}")

    # find lut
    lut_candidates = _find_file(atlas_dir, "*.txt") + _find_file(atlas_dir, "*.lut")
    if len(lut_candidates) == 1:
        lut_path = lut_candidates[0]
    elif len(lut_candidates) > 1:
        # resolve ambiguity
        filtered = [f for f in lut_candidates if 'LUT' in f or 'lut' in f]
        if len(filtered) == 1:
            lut_path = filtered[0]
        else:
            names = [os.path.basename(c) for c in lut_candidates]
            raise ValueError(f"Ambiguous LUTs found: {names}")
    else:
        raise FileNotFoundError(f"No LUT file found in custom directory: {atlas_dir}")
        
    return csv_path, lut_path

def _get_ordered_names(atlas_dir, file_map):
    """
    Attempts to read the strict region order from a LUT or order text file.
    Falls back to alphabetical sorting if no file exists.
    """
    txt_files = []

    # look for a LUT or order file, ignoring qc reports
    for root, dirs, files in os.walk(atlas_dir):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__')) and 'qc_report' not in d]
        for file in files:
            if file.endswith('.txt') and 'registry' not in file:
                txt_files.append(os.path.join(root, file))
    def file_priority(filepath):
        name = filepath.lower()
        if 'lut' in name: return 0
        if 'order' in name: return 1
        return 2
    txt_files.sort(key=file_priority)

    if txt_files:
        ordered_names = []
        with open(txt_files[0], 'r') as f:
            for line in f:
                parts = line.strip().split()
                # assuming standard LUT format (ID Name ...) or simple list (ID Name)
                if len(parts) >= 2:
                    name = parts[1]
                    if name in file_map and name not in ordered_names:
                        ordered_names.append(name)
        
        # append any stray files that exist in the directory but weren't in the text file
        for name in sorted(file_map.keys()):
            if name not in ordered_names:
                ordered_names.append(name)
        
        return ordered_names

    # legacy fallback: alphabetical
    return sorted(list(file_map.keys()))

def _find_subcortical_files(atlas_dir):
    """
    Internal: Scans directory for mesh files (.vtk preferred, then .gii).
    Returns a dictionary: {region_name: file_path}
    """
    
    def _scan_for_ext(directory, extension):
        """Recursively finds files with extension, ignoring junk folders."""
        candidates = []
        # check root
        candidates.extend(glob.glob(os.path.join(directory, f"*{extension}")))
        
        # check valid subdirectories
        try:
            subdirs = [
                os.path.join(directory, d) for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d)) 
                and not d.startswith(('.', '__'))
            ]
            for sd in subdirs:
                candidates.extend(glob.glob(os.path.join(sd, f"*{extension}")))
        except FileNotFoundError:
            pass
            
        return candidates

    # try finding VTK files
    vtk_files = _scan_for_ext(atlas_dir, ".vtk")
    if vtk_files:
        # map basename -> full path
        return {
            os.path.splitext(os.path.basename(f))[0]: f 
            for f in vtk_files
        }

    # if no VTKs, try GIfTI files
    gii_files = _scan_for_ext(atlas_dir, ".gii")
    if gii_files:
        # filter for '_surface.surf.gii' but fallback to just .gii if typical naming isn't found
        filtered_gii = [f for f in gii_files if '.surf.gii' in f]
        if not filtered_gii:
            filtered_gii = gii_files
            
        return {
            os.path.basename(f).split('.')[0]: f 
            for f in filtered_gii
        }

    raise FileNotFoundError(f"No .vtk or .gii mesh files found in {atlas_dir}")

def _find_tract_files(atlas_dir):
    """
    Internal: Scans directory for tractography files (.trk or .tck).
    Returns a dictionary: {tract_name: file_path}
    """
    
    def _scan_for_ext(directory, extension):
        """Recursively finds files with extension, ignoring junk folders."""
        candidates = []
        # check root
        candidates.extend(glob.glob(os.path.join(directory, f"*{extension}")))
        
        # check valid subdirectories
        try:
            subdirs = [
                os.path.join(directory, d) for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d)) 
                and not d.startswith(('.', '__'))
            ]
            for sd in subdirs:
                candidates.extend(glob.glob(os.path.join(sd, f"*{extension}")))
        except FileNotFoundError:
            pass
            
        return candidates

    if  os.path.isdir(atlas_dir):
        # scan for both .trk and .tck
        found_files = _scan_for_ext(atlas_dir, ".trk") + _scan_for_ext(atlas_dir, ".tck")
    elif os.path.isfile(atlas_dir):
        if atlas_dir.endswith((".trk", ".tck")):
            found_files = [atlas_dir]
    else:
        raise ValueError(f"Invalid atlas directory/file path: {atlas_dir}, no valid tck or trk file found.")

    
    if not found_files:
        raise FileNotFoundError(f"No .trk or .tck files found in {atlas_dir}")

    # map basename -> full path
    return {
        os.path.splitext(os.path.basename(f))[0]: f 
        for f in found_files
    }

