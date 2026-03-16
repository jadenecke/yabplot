from importlib.metadata import version, PackageNotFoundError

from .plotting import plot_cortical, plot_subcortical, plot_tracts, clear_tract_cache, plot_vertexwise
from .data import get_available_resources, get_atlas_regions
from .atlas_builder import build_cortical_atlas, build_subcortical_atlas
from .utils import load_vertexwise_mesh, project_vol2surf

try:
    __version__ = version("yabplot")
except PackageNotFoundError:
    __version__ = "unknown"