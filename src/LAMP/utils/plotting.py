"""Centralised plotting class/functions?
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def get_colormap(name, default='plasma', option=None):
    """Wrapper to return custom LAMP colourmaps"""
    if name.lower() == 'electron_beam':
        # Jet, but with white as first 5% of values
        orig_cm = mpl.colormaps['jet'].resampled(256)
        newcolors = orig_cm(np.linspace(0, 1, 256))
        white = np.array([256/256, 256/256, 256/256, 1])
        if option is not None:
            end_index = int((option/100)*256) # percentage to make white
        else:
            end_index = int((5/100)*256)
        newcolors[:end_index, :] = white
        colormap = ListedColormap(newcolors)
        # use set_under?
        # import matplotlib.colors as colors
        # palette = plt.get_cmap('jet')
        # palette.set_under('white', 0)  
        # norm = colors.BoundaryNorm(levels, ncolors=palette.N)
        # im = ax.imshow(Z, cmap=palette,
        #             norm=norm,
        #             aspect='auto', extent=[x0, x1, y0, y1])
    elif name in mpl.colormaps:
        # standard matplotlib map?
        colormap = mpl.colormaps[name]
    else:
        print(f'Warning get_colormap(), could not match colormap: {name}. Using {default}')
        colormap = mpl.colormaps[default]

    return colormap

def create_montage(images, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, num_rows=1, transpose=True, divider=True):
    """ num_rows currently only works for =1
    Also, downsampling counts not corrected!"""
    # m = num Y pixels, n = num X pixels, count = num images
    m, n, count = images.shape

    # TODO: Need to check ROIs are within pixel limits, or will cause errors below
    if x_roi:
        n = x_roi[1] - x_roi[0]
    else:
        x_roi = [0,n]
    if y_roi:
        m = y_roi[1] - y_roi[0]
    else:
        y_roi = [0,m]
    
    m = int(m /  y_downsample)
    n = int(n / x_downsample)
    
    num_cols = int(np.ceil(count / num_rows))
  
    if transpose:
        montage = np.zeros((num_rows * n, num_cols * m))
        x_locs = np.linspace(0, m * (count-1), count) + m / 2.0
    else:
        montage = np.zeros((num_rows * m, num_cols * n))
        x_locs = np.linspace(0, n * (count-1), count) + n / 2.0

    # need to make sure any down sampling fit array sizes
    if np.shape(images[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, 0])[0] > m:
        y_roi[1] = y_roi[1] - y_downsample
    if np.shape(images[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, 0])[1] > n:
        x_roi[1] = x_roi[1] - x_downsample

    max_val = np.max(images)

    image_id = 0
    for j in range(count):
        for k in range(num_rows):
            if image_id >= count:
                break
            if transpose:
                sliceJ = j * m
                sliceK = k * n
                montage[sliceK:sliceK + n, sliceJ:sliceJ + m] = images[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, image_id].T
                if divider and (j<(count-1)):
                    montage[:, (sliceJ + m - 2):(sliceJ + m + 2)] = max_val # 5 pixel wide divider
            else:
                sliceJ = j * n
                sliceK = k * m
                montage[sliceK:sliceK + m, sliceJ:sliceJ + n] = images[y_roi[0]:y_roi[1]:y_downsample, x_roi[0]:x_roi[1]:x_downsample, image_id]
                if divider and (j<(count-1)):
                    montage[:, (sliceJ + n - 2):(sliceJ + n + 2)] = max_val # 5 pixel wide divider
            image_id += 1

    return montage, x_locs

def plot_montage(images, x_roi=None, y_roi=None, axis=None, x_downsample=1, y_downsample=1, title='', num_rows=1, transpose=False, shot_labels=None, y_label=None, cb_label=None, vmin=None, vmax=None, colormap='plasma', colormap_option=None):
    """ images should be an [m, n, count] array of images, where m = Y size, n = X size
    ROI values in image pixels - probably easier to handle ROIs before this function?
    num_rows currently only works for =1"""

    # m = num Y pixels, n = num X pixels, count = num images
    m, n, count = images.shape

    if x_roi is not None:
        n = x_roi[1] - x_roi[0]
    else:
        x_roi = [0,n]
    if y_roi is not None:
        m = y_roi[1] - y_roi[0]
    else:
        y_roi = [0,m]

    montage, x_locs = create_montage(images, x_roi=x_roi, y_roi=y_roi, x_downsample=x_downsample, y_downsample=y_downsample, num_rows=num_rows, transpose=transpose)

    if vmax is None:
        vmax = np.percentile(montage, 99)
    if vmin is None:
        vmin = np.min(montage)

    if axis is None:
        if transpose:
            axis = np.arange(n)
        else:
            axis = np.arange(m)
    if transpose:
        yaxis = axis[x_roi[0]:x_roi[1]:x_downsample]
        xaxis = np.arange(montage.shape[1])
    else:
        yaxis = axis[y_roi[0]:y_roi[1]:y_downsample]
        xaxis = np.arange(montage.shape[1]) 

    if not shot_labels:
        shot_labels = np.arange(count)+1

    fig = plt.figure()
    ax = plt.gca()
    im = ax.pcolormesh(xaxis, yaxis, montage, vmin=vmin, vmax=vmax, shading='auto', cmap=get_colormap(colormap, option=colormap_option))
    ax.set_ylabel(y_label)
    ax.set_title(title, y=-0.2)
    divider = make_axes_locatable(ax)
    ax.set_xticks(x_locs)
    ax.set_xticklabels(shot_labels)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    if cb_label is not None:
        cb.set_label(cb_label, rotation=270)
    plt.tight_layout()

    return fig, ax