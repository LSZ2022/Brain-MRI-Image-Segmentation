import numpy as np
import numpy.typing as npt
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from scipy.ndimage import label
import torch


def largest_connected_component(img: np.ndarray) -> npt.NDArray[np.bool_]:
    # from medpy.filter.binary.largest_connected_component
    # Returns the largest connected component of the input 2d-image as binary mask
    labeled_array, num_features = label(img, None)  # type: ignore
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    return labeled_array == (np.argmax(component_sizes) + 1)


def dsc(ys_pred, ys_true, lcc=True):
    # Computes dice similarity coefficient between imput images
    if lcc and np.any(ys_pred):
        ys_pred = np.round(ys_pred).astype(int)
        ys_true = np.round(ys_true).astype(int)
        ys_pred = largest_connected_component(ys_pred)
    return np.sum(ys_pred[ys_true == 1]) * 2.0 / (np.sum(ys_pred) + np.sum(ys_true))


def log_images(xs: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, channel=1):
    images = []
    xs_np = xs[:, channel].cpu().numpy()
    ys_true_np = y_true[:, 0].cpu().numpy()
    ys_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(xs_np.shape[0]):
        image = gray2rgb(np.squeeze(xs_np[i]))
        image = outline(image, ys_pred_np[i], color=[255, 0, 0])
        image = outline(image, ys_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image: npt.NDArray[np.floating]):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max

    ret = np.tile((image[..., None] * 255).astype(np.uint8), reps=3)
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


def dataloader_worker_init(worker_id):
    np.random.seed(42 + worker_id)


