"""
This module provides functions for displaying images using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from .fileio import ensure_dir
from .colors import (color_palette, colors2plotly)
from pathlib import Path

# Useful to display pretty looking elements in Jupyter notebooks
from IPython.core.display import HTML


# Default color palette of this package.
PALETTE = color_palette("default", alpha=[0.8, 0.45, 0.15], 
                        mix_color="white")
PALETTE_CMAP_CONT_BRG = color_palette(PALETTE,
                                      mix_color="white", as_cmap=True)
PALETTE_CMAP_CONT_RBG = color_palette([PALETTE[1],PALETTE[0], PALETTE[2]],
                                      mix_color="white", as_cmap=True)
PALETTE_CMAP_CONT_BR = color_palette(PALETTE[:2],
                                     mix_color="black", as_cmap=True)
PALETTE_CMAP_CONT_RG = color_palette(PALETTE[1:3],
                                     mix_color="black", as_cmap=True)

PALETTE_CMAP = color_palette("default",
                             mix_color="white", as_cmap=True, n_cmap=3)
PALETTE_PLOTLY = colors2plotly(PALETTE)
# PALETTE_RGB = [PALETTE[1], PALETTE[2], PALETTE[0]]


def show_header(title=None, 
                subtitle=None, 
                width=9,  # str or (float in inches)
                
                # Default title style
                fontsize=24, 
                color=None,
                kwargs={},
                
                # Default subtitle style
                subtitle_fontsize=16, 
                subtitle_color=None,
                subtitle_kwargs={},
                ):
    """Displays a header with a title and a subtitle."""
        
    if width is None:
        width_str = "9in"
    elif isinstance(width, str):
        width_str = width
    elif isinstance(width, (int, float)):
        width_str = "%din" % width
    else:
        raise ValueError("Invalid type for width: %s" % type(width))
    
    title_style = dict()
    title_style.update(kwargs)
    title_style.setdefault("width", width_str)
    title_style.setdefault("text-align", "center")
    title_style.setdefault("font-size", "%dpx" % fontsize)
    title_style.setdefault("font-weight", "bold")
    title_style.setdefault("color", color or "#333333")
    title_style.setdefault("border", "none")
    title_style.setdefault("margin", "30px 0 0 0")
    title_style.setdefault("vertical-align", "middle")
    # title_style.setdefault("padding", "0px")
    # title_style.setdefault("background-color", "transparent")
    
    subtitle_style = dict()
    subtitle_style.update(subtitle_kwargs)
    subtitle_style.setdefault("width", width_str)
    subtitle_style.setdefault("text-align", "center")
    subtitle_style.setdefault("font-size", "%dpx" % subtitle_fontsize)
    #subtitle_style.setdefault("font-weight", "bold")
    subtitle_style.setdefault("color", subtitle_color or "#999999")
    subtitle_style.setdefault("border", "none")
    subtitle_style.setdefault("margin", "0 0 0 0")
    subtitle_style.setdefault("padding", "0px")
    # subtitle_style.setdefault("background-color", "transparent")
    
    # Convert styles to string
    def style2str(style):
        return "; ".join(["%s:%s" % (key, value) for key, value in style.items() if value is not None])  
    

    if title is not None:
        html = """
                <input
                type="text"
                style="%s"
                value="%s"
                />
            """
        html = html % (style2str(title_style), title)
        display(HTML(html));
    
    if subtitle is not None:
        html = """
                <input
                type="text"
                style="%s"
                value="%s"
                />
            """
        html = html % (style2str(subtitle_style), subtitle)
        display(HTML(html));


# Set default color palett
def setup_plotting(palette=None, no_seaborn=False, **kwargs):
    """
    Adjusts the default plotting settings:
    - Sets the color palette
    """
    if not no_seaborn:
        import seaborn as sns
        sns.set_style("whitegrid")
    
    import matplotlib.pyplot as plt
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)
    plt.rcParams["axes.titleweight"] = 'bold'
    plt.rcParams["grid.linestyle"] = '-'
    plt.rcParams["grid.alpha"] = 0.4
    
    
    #plt.rcParams["figure.figsize"] = (5, 3)
    #plt.rcParams["figure.dpi"] = 300
    for key, value in kwargs.items():
        plt.rcParams[key] = value
        
    # For selectable text in PDFs
    matplotlib.rc("pdf", fonttype=42)
    

def display_image(image=None, scale=None):
    """Displays an image using IPython capabilities."""
    from IPython.display import display
    import PIL.Image
    if isinstance(image, (str, Path)):
        image = PIL.Image.open(image)
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    if scale is not None:
        image = image.resize((int(image.width * scale), 
                              int(image.height * scale)))
    display(image)
    

def draw_image(image, **kwargs):
    """Alias for show_image()"""
    return show_image(image, **kwargs)


def show_image(image, title=None, 
               ax=None, shape=None, 
               normalize=False,
               normalize_stretch=None,
               dpi=100, 
               suppress_info=False,
               background_color=(0.93, 0.93, 0.93),
               frame=False, 
               frame_color=(0.8,)*4,
               frame_width=2,
               axes_frame=True,
               show_axes=False,
               box_aspect=None,
               save_kwargs={},
               ):
    """Displays an image using matplotlib capabilities.

    Args:
        image: The image to display.
        title: The title of the image.
        ax:    The image axis to use. If None, a new figure is created.
        shape: The shape of the canvas. If None, the shape is inferred 
               from the image.
        normalize: If True, grayscale images are normalized so that the 
                minimal and maximal values are mapped to black and white, 
                respectively. If normalize is a 2-tuple, the provided values
                will be used for normalization (vmin, vmax = normalize). If 
                False or None, the images are displayed using the full range 
                of the current data type. If "stretch", the contrast is
                stretched using the 1% and 99% percentiles of the intensity.
        normalize_stretch: If not None, the contrast is stretched using the
                provided percentiles of the intensity. Sets normalize to True.
        frame: If True, a frame is drawn around the image (if the image
               is smaller than the canvas).
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
    """
    height, width = image.shape[:2]
    
    if ax is None:
        # Create a figure of the right size with 
        # one axis that takes up the full figure
        figsize = width / float(dpi), height / float(dpi)
        if figsize[0]>9 and True:
            figsize = (9, figsize[1] * 9 / figsize[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

    # If the image is grayscale, use the gray colormap.
    cmap = "gray" if len(image.shape) == 2 else None

    vmin, vmax = None, None
    if normalize_stretch is not None:
        if isinstance(normalize_stretch, (float, int)):
            normalize_stretch = (normalize_stretch, 100-normalize_stretch)
        vmin, vmax = np.percentile(image, normalize_stretch)
    elif not normalize:
        # imshow normalization is on by default!
        # If one wants to disable it, one must set vmin and vmax.
        dtype = image.dtype
        if dtype == np.uint8:
            vmin, vmax = 0, 255
        elif dtype == np.uint16:
            vmin, vmax = 0, 65535
        elif dtype in (np.float32, np.float64, float, bool):
            vmin, vmax = 0, 1.0
        else:
            assert False, "Unsupported data type: %s" % dtype
    elif isinstance(normalize, (tuple, list)) and len(normalize) == 2:
        vmin, vmax = normalize
            
    ax.imshow(image, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)

    title = "" if title is None else title
    if not suppress_info:
        title += "\n" if title else ""
        title += "(%s, %s)" % ("x".join(map(str, image.shape)), image.dtype)
    if title:
        ax.set_title(title)
    
    # Use fixed axis limits so that
    # the images are shown at scale.
    if shape is not None:
        ax.set_xlim([0, shape[1]])
        ax.set_ylim([shape[0], 0])

    # Use a background color to better see that the images are transparent.
    if background_color is None and (not show_axes):
        ax.axis("off")
    ax.set_facecolor(background_color)
    if not show_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_frame_on(axes_frame) 
    ax.set_anchor("N")
    if box_aspect:
        ax.set_box_aspect(box_aspect)
        # Strange hack required if box_aspect is set, but only sometimes.
        ax.update_datalim([[ax.get_xlim()[1], ax.get_ylim()[1]]])

    h, w = image.shape[:2]
    # Draw frame if:
    if frame:
        rect = patches.Rectangle((0, 0), w, h, 
                                linewidth=frame_width, 
                                edgecolor=frame_color, 
                                facecolor='none')

        # Add the rectangle patch to the plot
        ax.add_patch(rect)  
        
    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
        

def show_image_pair(image1, image2, 
                    title1=None, title2=None, 
                    normalize=True, 
                    dpi=None,
                    figsize=(6, 5),
                    shape="largest",
                    box_aspect=None,
                    frame=True,
                    save_kwargs={},
                    **kwargs):
    """Displays a pair of images side-by-side.

    Args:
        image1: The first image.
        image2: The second image.
        title1: The title of the first image.
        title2: The title of the second image.
        normalize: If True, grayscale images are normalized.
        dpi:    The DPI of the figure.
        frame:  If True, a frame is drawn around the images 
                (if the images are smaller than the canvas).
                Set "forced" to force a frame.
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
        kwargs:  Additional keyword arguments passed to show_image().
    """
    # This converts PIL images to numpy arrays.
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    max_shape = (max(image1.shape[0], image2.shape[0]),
                 max(image1.shape[1], image2.shape[1]))
    
    if dpi is not None:
        # Overrides figsize
        figsize = (max_shape[0]/dpi, max_shape[1]/dpi)

    if shape=="largest":
        shape = max_shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    draw_frame1 = ((frame=="forced") or 
                   (frame and shape is not None and shape!=image1.shape[:2]))
    draw_frame2 = ((frame=="forced") or 
                   (frame and shape is not None and shape!=image2.shape[:2]))
    
    show_image(image1, title1, ax1, 
               normalize=normalize, 
               shape=shape, 
               box_aspect=box_aspect,
               frame=draw_frame1,
               **kwargs)
    show_image(image2, title2, ax2, 
               normalize=normalize, 
               shape=shape, 
               box_aspect=box_aspect,
               frame=draw_frame2,
               **kwargs) 
    fig.tight_layout()
    
    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
    
    plt.show()


def show_image_chain(images, **kwargs):
    """Displays a list of images. Equivalent to show_image_grid(..., ncols=-1).
    """
    kwargs.setdefault("ncols", -1)
    show_image_grid(images, **kwargs)


def show_image_grid(images, titles=None, 
                    ncols=3, 
                    scale=4.0, 
                    figsize=None, 
                    shape="largest",
                    dpi=100,
                    suppress_info=False,
                    normalize=True,
                    box_aspect=None,
                    frame=True,
                    header=None,
                    header_kwargs={},
                    save_kwargs={},
                    **kwargs):
    """Displays a grid of images. The width of the grid is determined by ncols.

    Args:
        images: A list of images, a generator of images, or a dictionary of
                titles and images.
        titles: A list of titles for the images (optional). If images are 
                provided as a dictionary, the titles are inferred from the
                dictionary keys.
        ncols:  The number of columns in the grid. If ncols=-1, the number of 
                columns is set to the number of images.
        scale:  The scale of the figure.
        figsize: The size of the figure. (Overrides scale.)
        shape:  The shape of the image canvas. Default: "largest". If 
                "largest", the shape is set to the largest image, and all 
                images are shown at the same scale. If None, the shape is 
                inferred from the images. 
        dpi:    The DPI of the figure. Default: 100. Only relevant if
                figsize is not set. The real DPI will not be exactly 
                the same as the requested DPI.
        suppress_info: If True, the image shape and data type are not shown
                in the title.
        normalize: If True, grayscale images are normalized so that the 
                minimal and maximal values are mapped to black and white, 
                respectively. If normalize is a 2-tuple, the provided values
                will be used for normalization (vmin, vmax = normalize). If 
                False or None, the images are displayed using the full range 
                of the current data type. 
        box_aspect: If not None, the aspect ratio of the image is fixed.
                Useful for images with different aspect ratios.
        frame:  If True, a frame is drawn around the images (if the images
                are smaller than the canvas). Set "forced" to force a frame.
        header: If not None, a header is displayed above the images.
        header_kwargs: Additional keyword arguments passed to show_header().
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
        kwargs:  Additional keyword arguments passed to show_image().
        
    Usage:
        show_image_grid([image1, image2, image3]) 
        show_image_grid({title1: image1, title2: image2, title3: image3})
    """
    if not images:
        return
    
    # Manage input types
    import types
    if isinstance(images, types.GeneratorType):
        images = list(images)
    elif isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    if isinstance(titles, types.GeneratorType):
        titles = list(titles)
    elif titles is None:
        titles = [None] * len(images)
    
    images = [None if img is None else np.asarray(img) for img in images]
    assert titles is None or (len(images) == len(titles))
    has_titles = any(titles)

    # Number of rows and columns
    if ncols == -1:
        ncols = len(images)
        nrows = 1
    else:
        ncols = min(len(images), ncols)
        nrows = int(np.ceil(len(images) / ncols))

    # Manage shape
    h_max, w_max = np.vstack([ img.shape[:2] for img in images if img is not None ]).max(axis=0)
    
    # Height of title in inches
    h_title = (has_titles*1 + (not suppress_info)*0.5)*scale
    h_fig = (h_max / dpi * scale + h_title)*nrows
    w_fig = (w_max / dpi * scale)*ncols

    if shape == "largest":
        shape = (h_max, w_max)

    # Figure size
    if figsize is None:
        figsize = (w_fig, h_fig)
    # Limit the figure size to 9 inches in width.
    # this is a good size for printing
    if figsize[0]>9:
        figsize = (9, figsize[1] * 9 / figsize[0])
        
    if header:
        show_header(title=header,
                    width=figsize[0],
                    fontsize=24,
                    kwargs=header_kwargs)

    # Create the figure
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=figsize,
                             squeeze=False)
    
    for ax, image, title in zip(axes.flat, images, titles):    
        if image is None:
            ax.axis("off")
            continue        
        draw_frame = ((frame=="forced") or 
                      (frame and shape is not None and shape!=image.shape[:2]))
        show_image(image, title=title, ax=ax, 
                   suppress_info=suppress_info, 
                   normalize=normalize,
                   box_aspect=box_aspect,
                   shape=shape,
                   frame=draw_frame,
                   **kwargs)
        
    # Disable grid axes that are not used
    for i in range(len(images), len(axes.flat)):
        axes.flat[i].axis("off")
    fig.tight_layout()
    
    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
    
    plt.show()


def save_figure(fig=None, path="figure.pdf", **kwargs):
    if fig is None:
        fig = plt.gcf()
    kwargs.setdefault("transparent", True)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("dpi", 300)
    path = Path(path)
    ensure_dir(path.parent)
    plt.savefig(path, **kwargs)



def plot_decision_boundary(clf, X, y, 
                           X_test=None, y_test=None, 
                           n_steps=1000, data=None, ax=None,
                           legend="grouped"):
    """
    Plot the decision boundaries of a classifier with two features.
    You don't need to understand the details of this function.
    
    Args:
        clf: The classifier to plot.
        X: The features of the dataset.
        y: The labels of the dataset.
        X_test: (optional) The features of the test dataset
        y_test: (optional) The labels of the test dataset
        n_steps: Parameter controlling the resolution of the plot.
        ax: The axis to plot on. If None, a new figure is created.
        data: Data structure provided by sklearn.datasets.load_iris().
    """
    assert isinstance(X, pd.DataFrame)
    if ax is None:
        _, ax = plt.subplots()
    x1, x2 = X.iloc[:, 0], X.iloc[:, 1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, n_steps),
        np.linspace(x2_min, x2_max, n_steps)
    )
    zz = clf.predict(pd.DataFrame(np.c_[xx1.ravel(), xx2.ravel()],
                                  columns=X.columns))
    zz = zz.reshape(xx1.shape)
    ax.contourf(xx1, xx2, zz, cmap=plt.cm.RdYlBu, alpha=0.4) 
    handles = []
    for i, color in zip(range(3), "ryb"):
        group = []
        label = data.target_names[i] if data is not None else ("Class %d" % i)
        h = ax.scatter(
            x1[y == i],
            x2[y == i],
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.25,
            s=20,
        )
        group.append(h)
        if X_test is not None and y_test is not None:
            label = ((data.target_names[i] + " (test)") if data is not None 
                    else ("Class %d (test)" % i))
            label = None
            h = ax.scatter(
                X_test.iloc[:, 0][y_test == i],
                X_test.iloc[:, 1][y_test == i],
                color=color,
                label=label,
                edgecolor="black",
                linewidth=0.25,
                s=20,
                marker="^",
            )
            group.append(h)
        handles.append(tuple(group))
    ax.set_xlabel(x1.name)
    ax.set_ylabel(x2.name)
    
    # Add legend with different markers for test data, if present.
    # https://matplotlib.org/stable/users/explain/axes/legend_guide.html
    if legend == "grouped":
        from matplotlib.legend_handler import HandlerTuple
        suffix = ""
        if data is not None:
            labels = data.target_names
        else:
             labels = ["Class %d" % i for i in range(3)]
            
        if X_test is not None and y_test is not None:
            suffix = " (train/test)"
        ax.legend(handles=handles, 
                labels=[name+suffix for name in labels],
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fontsize="small")
    elif legend == "simple":
        ax.legend()
    elif legend in ("none", False, None):
        ax.legend().remove()
    else:
        raise ValueError("Invalid value for 'legend'.")