
import numpy as np
import cv2
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE



def to_superscript(number):
    superscript_map = {
    "0": "\u2070",
    "1": "\u00B9",
    "2": "\u00B2",
    "3": "\u00B3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079"}
    return ''.join(superscript_map.get(char, char) for char in str(number))
    

def write_powers(power):
    base = 10
    if power < 0:
        # For negative powers, use the superscript minus sign (⁻) followed by the power
        power_str = "\u207B" + to_superscript(abs(power))
    else:
        power_str = to_superscript(power)
    return f"{base}{power_str}"


# Ensure required packages are installed: pip install umap-learn opencv-python matplotlib scikit-learn

def scale_to_01_range(x):
    """Scale values between 0 and 1."""
    value_range = np.max(x) - np.min(x)
    return (x - np.min(x)) / value_range if value_range != 0 else x

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    """Compute image positions on the plot."""
    image_height, image_width, _ = image.shape
    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * (1 - y)) + offset  # Mirror y-axis

    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def scale_image(image, max_image_size):
    """Resize images while maintaining aspect ratio."""
    image_height, image_width, _ = image.shape
    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    return cv2.resize(image, (image_width, image_height))

def visualize_dim_reduction_images(data, images, save_path, method="UMAP", dpi=300, plot_size=1000, max_image_size=100):
    """
    Perform dimensionality reduction (UMAP or t-SNE) and visualize images.
    
    Parameters:
    - data: (N, D) numpy array of feature vectors.
    - images: List of image paths corresponding to each data point.
    - name: Output filename.
    - method: Choose between "UMAP" and "tSNE".
    - dpi: Resolution of the output plot.
    - plot_size: Size of the final visualization.
    - max_image_size: Max image size on the plot.
    """

    # Perform dimensionality reduction
    if method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedding = reducer.fit_transform(data)
    else:
        raise ValueError("Invalid method! Choose 'UMAP' or 'tSNE'.")

    # Normalize the reduced output to [0,1] range
    tx = scale_to_01_range(embedding[:, 0])
    ty = scale_to_01_range(embedding[:, 1])

    # Set up the visualization plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset
    plot_canvas = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # Place images on the scatter plot
    for i in range(len(images)):
        x, y = tx[i], ty[i]
        try:
            image = cv2.imread(images[i])
            image = scale_image(image, max_image_size)
            tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
            plot_canvas[tl_y:br_y, tl_x:br_x, :] = image  # Place image on plot
        except AttributeError:
            pass  # Ignore images that fail to load

    # Plot and save figure
    fig = plt.figure(figsize=(80, 50), dpi=100)
    plt.imshow(plot_canvas[:, :, ::-1])
    plt.axis("off")
    plt.title(f"{method} plot", fontsize=50)
    plt.savefig(save_path, dpi=dpi)
    plt.show()


