import os
import numpy as np
import pyvista as pv
import warnings

def get_shading_preset(style_name):
    """
    Returns a dictionary of lighting parameters for pyvista.add_mesh.

    Styles:
    - 'default': Balanced, no shine.
    - 'matte':   (Soft) High ambient, low contrast. Good for reading atlas colors.
    - 'sculpted':(Hard) Stronger shadows, higher contrast. Good for showing anatomy.
    - 'glossy':  (Shiny) Wet/Plastic look with specular highlights.
    """
    presets = {
        'default': {
            'lighting': True,
            'specular': 0.0,
            'ambient': 0.65,
            'diffuse': 0.4,
            'specular_power': 15
        },
        # very bright shadows
        'matte': {
            'lighting': True,
            'specular': 0.0,
            'ambient': 0.75,
            'diffuse': 0.2,
            'specular_power': 0
        },
        # slight shime, dark shadows, strong directional light
        'sculpted': {
            'lighting': True,
            'specular': 0.05,
            'ambient': 0.4,
            'diffuse': 0.6,
            'specular_power': 10
        },
        # strong shine, sharp highlights
        'glossy': {
            'lighting': True,
            'specular': 0.3,
            'ambient': 0.4,
            'diffuse': 0.6,
            'specular_power': 30
        },
        # flat 2D
        'flat': {
            'lighting': False,
            'ambient': 1.0,
            'diffuse': 0.0,
            'specular': 0.0
        }
    }

    if style_name not in presets:
        print(f"Warning: Style '{style_name}' not found. Using 'default'. Options: {list(presets.keys())}")
        return presets['default']

    return presets[style_name]

def get_view_configs(view_names):
    all_views = {
        'left_lateral':  {'pos': (-1, 0, 0), 'up': (0, 0, 1), 'side': 'L'},
        'right_lateral': {'pos': (1, 0, 0),  'up': (0, 0, 1), 'side': 'R'},
        'left_medial':   {'pos': (1, 0, 0),  'up': (0, 0, 1), 'side': 'L'},
        'right_medial':  {'pos': (-1, 0, 0), 'up': (0, 0, 1), 'side': 'R'},
        'superior':      {'pos': (0, 0, 1),  'up': (0, 1, 0), 'side': 'both'},
        'inferior':      {'pos': (0, 0, -1), 'up': (0, 1, 0), 'side': 'both'},
        'anterior':      {'pos': (0, 1, 0),  'up': (0, 0, 1), 'side': 'both'},
        'posterior':     {'pos': (0, -1, 0), 'up': (0, 0, 1), 'side': 'both'}
    }
    if view_names is None: return all_views
    return {k: all_views[k] for k in view_names if k in all_views}

def prepare_plotter(ax, display_type, sel_views, layout, figsize):
    """
    consolidates the logic for setting up the matplotlib figure/axis
    and determining the optimal pyvista window size.
    """
    import matplotlib.pyplot as plt

    # handle legacy pixel values (safety check)
    if figsize is not None:
        if figsize[0] > 100 or figsize[1] > 100:
            import warnings
            warnings.warn(
                f"figsize {figsize} seems to be in pixels (legacy). "
                f"automatically converting to inches (dividing by 100)."
            )
            figsize = (figsize[0] / 100, figsize[1] / 100)

    # handle case where user provides their own matplotlib axis
    if ax is not None:
        display_type = 'matplotlib'
        # detect the aspect ratio of the user's axis to prevent distortion
        try:
            bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
            ax_w, ax_h = bbox.width, bbox.height
            # use 1500px as a high-quality base for the render
            scale_f = 1500 / max(ax_w, ax_h)
            figsize = (int(ax_w * scale_f), int(ax_h * scale_f))
        except:
            # fallback if axis geometry isn't available yet
            figsize = (1000, 600)

    # handle automatic matplotlib figure creation
    if display_type == 'matplotlib' and ax is None:
        if figsize is not None:
            # respect explicit user figsize (inches)
            fig_w, fig_h = figsize
            figsize_px = (int(fig_w * 200), int(fig_h * 200))
        else:
            # calculate optimal figsize based on views
            n_v = len(sel_views)
            if layout: nr, nc = layout
            else:
                if n_v <= 1: nr, nc = 1, 1
                elif n_v <= 4: nr, nc = 1, n_v
                elif n_v <= 6: nr, nc = 2, 3
                else: nr, nc = int(np.ceil(n_v/4)), 4
            # base unit: 3.5x3 inches per subplot
            fig_w, fig_h = nc * 3.5, nr * 3.0
            figsize_px = (int(fig_w * 200), int(fig_h * 200))

        # use 150 dpi for manageable notebook display size
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        figsize = figsize_px

    # handle non-matplotlib modes with provided figsize
    elif figsize is not None and display_type != 'matplotlib':
        # convert user inches to pyvista pixels
        figsize = (int(figsize[0] * 200), int(figsize[1] * 200))

    return ax, display_type, figsize

def setup_plotter(sel_views, layout, figsize, display_type, needs_bottom_row=True):
    """
    initializes the pyvista plotter with the correct shape and window size.
    """
    n = len(sel_views)
    if layout is None:
        if n <= 1: base_layout = (1, 1)
        elif n <= 4: base_layout = (1, n)
        elif n <= 6: base_layout = (2, 3)
        else: base_layout = (int(np.ceil(n/4)), 4)
    else: base_layout = layout

    nrows_geo, ncols_geo = base_layout

    # calculate cell-based window size if not provided
    cell_w, cell_h = 300, 250

    if needs_bottom_row and display_type != 'matplotlib':
        nrows, ncols = nrows_geo + 1, ncols_geo
        groups = [(nrows - 1, slice(0, ncols))]
        row_weights = [1.0] * nrows_geo + [0.2]
        win_w = ncols * cell_w
        win_h = (nrows_geo * cell_h) + (cell_h * 0.2)
    else:
        nrows, ncols = nrows_geo, ncols_geo
        groups = None
        row_weights = None
        win_w = ncols * cell_w
        win_h = nrows * cell_h

    actual_size = figsize if figsize is not None else (int(win_w), int(win_h))

    # ensure high quality by forcing off-screen for non-interactive modes
    plotter = pv.Plotter(shape=(nrows, ncols), groups=groups, row_weights=row_weights,
                         off_screen=(display_type in ['object', 'static', 'matplotlib', 'pyvista']),
                         window_size=actual_size, border=False)
    plotter.set_background('white')
    return plotter, ncols, nrows

def add_context_to_view(plotter, bmesh, view_side, alpha, color, **kwargs):
    """
    Adds context mesh. Lighting parameters are passed via **kwargs.
    """
    if not bmesh: return
    for h, mesh in bmesh.items():
        if (view_side == 'L' and h == 'R') or (view_side == 'R' and h == 'L'): continue
        plotter.add_mesh(mesh, color=color, opacity=alpha,
                         smooth_shading=True, show_edges=False,
                         **kwargs)

def set_camera(plotter, view_cfg, zoom=1.0, distance=200):
    plotter.camera.position = tuple(p * distance for p in view_cfg['pos'])
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = view_cfg['up']
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(zoom)

def add_colorbars(plotter, mappers, titles, nrows, figsize):
    """
    Adds unified, cleanly formatted colorbars to the bottom row of the plot.
    """
    plotter.subplot(nrows - 1, 0)

    if figsize is None:
        figsize = plotter.window_size

    valid_mappers = [m for m in mappers if m is not None]
    valid_titles = [t for m, t in zip(mappers, titles) if m is not None]
    num_bars = len(valid_mappers)

    if num_bars == 0:
        return

    # calculate dynamic sizing based on window aspect ratio
    width_px, height_px = figsize
    aspect_ratio = width_px / height_px
    cb_width = 0.35 if aspect_ratio > 1.5 else 0.60
    pos_x = (1.0 - cb_width) / 2.0
    cb_height = 0.2

    if num_bars == 1:
        positions_y = [0.15]
        cb_height = 0.4
    elif num_bars == 2:
        positions_y = [0.5, 0.01]
    else:
        positions_y = np.linspace(0.35, 0.05, num_bars)

    for i, (mapper, title) in enumerate(zip(valid_mappers, valid_titles)):
        plotter.add_scalar_bar(
            mapper=mapper, title=title, vertical=False,
            position_x=pos_x, position_y=positions_y[i],
            height=cb_height, width=cb_width, color='black',
            title_font_size=7, label_font_size=7, n_labels=3, fmt="%g"
        )

def finalize_plot(plotter, export_path, display_type, ax=None, cbar_info=None, cbar_kwargs=None):
    """
    finalizes the plot by rendering, saving, and returning the requested object.
    """

    # matplotlib mode: returns (fig, ax) or just ax
    if display_type == 'matplotlib':
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # take screenshot of the 3d scene
        plotter.render()
        img = plotter.screenshot(transparent_background=True, return_img=True, scale=5)
        plotter.close()

        # convert to matplotlib figure/axis
        ret_ax = ax
        ret_fig = None
        if img is not None and ax is not None:
            ax.imshow(img)
            ax.axis('off')
            ret_fig = ax.get_figure()

        # add matplotlib colorbar(s)
        if cbar_info and ax is not None:
            try:
                import cmocean
            except ImportError:
                cmocean = None

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)

            for i, info in enumerate(cbar_info):
                cmap_name = info.get('cmap', 'coolwarm')
                vminmax = info.get('vminmax', [0, 1])
                title = info.get('title', '')
                vmin = vminmax[0] if vminmax[0] is not None else 0
                vmax = vminmax[1] if vminmax[1] is not None else 1

                # attempt to find colormap (including cmocean)
                cmap_obj = None
                try:
                    cmap_obj = plt.get_cmap(cmap_name)
                except:
                    if cmocean and hasattr(cmocean.cm, str(cmap_name)):
                        cmap_obj = getattr(cmocean.cm, str(cmap_name))
                    else:
                        cmap_obj = plt.get_cmap('coolwarm')

                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                sm.set_array([])

                cax_pad = 0.3 if i == 0 else 0.8
                cax_container = divider.append_axes("right", size="1%", pad=cax_pad)
                cax_container.axis('off')
                cb_height = 0.4
                cb_bottom = (1.0 - cb_height) / 2.0
                cax = cax_container.inset_axes([0, cb_bottom, 1.0, cb_height])

                cbar_opts = {}
                cbar_opts.update(cbar_kwargs or {})
                cbar = plt.colorbar(sm, cax=cax, **cbar_opts)

                if title:
                    cbar.set_label(title, fontsize=7)
                cbar.ax.tick_params(labelsize=7)

        # save the full matplotlib figure if path provided
        if export_path and ret_fig is not None:
            ret_fig.savefig(export_path, bbox_inches='tight', dpi=300, transparent=True)

        return ret_ax

    # interactive mode: opens trame browser viewer
    elif display_type == 'interactive':
        # saving here captures only the 3d scene
        if export_path:
            plotter.screenshot(export_path, transparent_background=True, scale=5)
        return plotter.show(jupyter_backend='trame')

    # pyvista mode: returns static jupyter widget (legacy behavior)
    elif display_type == 'pyvista':
        if export_path:
            plotter.screenshot(export_path, transparent_background=True, scale=5)
        out = plotter.show(jupyter_backend='static')
        plotter.close()
        return out

    # object mode: returns raw plotter for manual control
    elif display_type == 'object':
        # user handles rendering/saving themselves
        return plotter

    # fallback: cleanup and return nothing
    else:
        if export_path:
            plotter.screenshot(export_path, transparent_background=True, scale=5)
        plotter.close()
        return None
