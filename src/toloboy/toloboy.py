from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

####################
#### functions #####
####################

############################################
#### Color space transformers functions ####
############################################


def RGB2LAB(
    r0: npt.NDArray[np.uint8],
    g0: npt.NDArray[np.uint8],
    b0: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    convert RGB to the personal LAB (LAB2)
    the input R,G,B,  must be 1D from 0 to 255
    the outputs are 1D  L [0 1], a [-1 1] b [-1 1]

    Expeceted arguments are flattened array for each
    channel (RGB).

    Parameters
    ----------
    R0 : np.uint8
        1D numpy array containing the R channel from an RGB image array.
    G0 : np.uint8
        1D numpy array containing the G channel from an RGB image array.
    B0 : np.uint8
        1D numpy array containing the B channel from an RGB image array.

    Returns
    -------
    tuple[np.uint8, np.uint8, np.uint8]
        return a tuple of size 3 containing the 1D channels (L, A, B)
    """

    R = r0 / 255
    G = g0 / 255
    B = b0 / 255

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    X = 0.449 * R + 0.353 * G + 0.198 * B
    Z = 0.012 * R + 0.089 * G + 0.899 * B

    L = Y
    A = (X - Y) / 0.234
    B = (Y - Z) / 0.785

    return L, A, B


def LAB22RGB(
    L: npt.NDArray[np.uint8],
    a: npt.NDArray[np.uint8],
    b: npt.NDArray[np.uint8],
) -> tuple[np.uint8, np.uint8, np.uint8]:
    """
    LAB22RGB _summary_

    onvert the personal LAB (LAB2)to the RGB
    the input L,a,b,  must be 1D L [0 1], a [-1 1] b [-1 1]
    the outputs are 1D  R g B [0 255]

    Parameters
    ----------
    L : np.uint8
        1D numpy array containing the L channel from an LAB image array.
    a : np.uint8
        1D numpy array containing the A channel from an LAB image array.
    b : np.uint8
        1D numpy array containing the B channel from an LAB image array.

    Returns
    -------
    tuple[np.uint8, np.uint8, np.uint8]
        return a tuple of size 3 containing the 1D channels (R, G. B)

    """
    # L, a, b = [array.reshape(-1, 1) for array in [L, a, b]]
    a11 = 0.299
    a12 = 0.587
    a13 = 0.114
    a21 = 0.15 / 0.234
    a22 = -0.234 / 0.234
    a23 = 0.084 / 0.234
    a31 = 0.287 / 0.785
    a32 = 0.498 / 0.785
    a33 = -0.785 / 0.785

    aa = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    C0 = np.zeros((L.shape[0], 3))
    C0[:, 0] = L[:, 0]
    C0[:, 1] = a[:, 0]
    C0[:, 2] = b[:, 0]
    C = np.transpose(C0)

    X = np.linalg.inv(aa).dot(C)
    X1D = np.reshape(X, (X.shape[0] * X.shape[1], 1))
    p0 = np.where(X1D < 0)
    X1D[p0[0]] = 0
    p1 = np.where(X1D > 1)
    X1D[p1[0]] = 1
    Xr = np.reshape(X1D, (X.shape[0], X.shape[1]))

    Rr = Xr[0][:]
    Gr = Xr[1][:]
    Br = Xr[2][:]

    R = np.uint(np.round(Rr * 255))
    G = np.uint(np.round(Gr * 255))
    B = np.uint(np.round(Br * 255))

    return R, G, B


def from_LAB_to_RGB_img(
    L: npt.NDArray[np.uint8],
    AB: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """
    Takes the L and AB channels retunred from the transformation and
    convert the image to RGB colorspace.

    retuns a 3d np.

    Parameters
    ----------
    L : _type_
        _description_
    AB : _type_
        _description_

    Returns
    -------
    np.ndarray :
    A 3D numpyarray containing the RGB image.
    """
    x_dim, y_dim = L.shape[0], L.shape[1]
    predicted_RGB = np.zeros((x_dim, y_dim, 3), dtype=np.uint8)
    AB = np.squeeze(AB)
    # print(f"Shape o AB in conversion is {AB.shape}")
    a0, b0 = AB[:, :, 0], AB[:, :, 1]

    # print(f"{np.squeeze(L).shape}, {a0.shape}, {b0.shape}")

    Rr, Gr, Br = LAB22RGB(L.reshape(-1, 1), a0.reshape(-1, 1), b0.reshape(-1, 1))

    predicted_RGB[:, :, 0] = np.reshape(Rr, (x_dim, y_dim))
    predicted_RGB[:, :, 1] = np.reshape(Gr, (x_dim, y_dim))
    predicted_RGB[:, :, 2] = np.reshape(Br, (x_dim, y_dim))

    return predicted_RGB


def plot_multiple_imgs(
    orig_img: npt.NDArray[np.uint8],
    imgs_ls: list[npt.NDArray[np.uint8] | Any],
    with_orig: bool = True,  # noqa: FBT001, FBT002
    col_title: list[str] | Any = None,
    img_size: int = 10,
    font_s: int = 12,
    **imshow_kwargs: Any,
) -> None:
    """
    Plot a reference image with a list of other images.

    Usefull to compare a reference image with
    a list of additional images after transformation,
    prediction, etc.

    Parameters
    ----------
    orig_img : npt.NDArray[np.uint8]
        The reference image.
    imgs_ls : list[npt.NDArray[np.uint8]  |  Any]
        A list of one or multiples images.
    with_orig : bool, optional
        NOTE: need to check what this does in practice dso I can maybe remove it, by default True
    col_title : list[str] | Any, optional
        List of titles to ad to the images , by default "None"
    img_size : int, optional
        size of the canvas, by default 10
    font_s : int, optional
        font size for the titles, by default 12

    Returns
    -------
    None
        return a plot of multiples images.
    """

    if not isinstance(imgs_ls[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs_ls = [imgs_ls]

    num_rows = len(imgs_ls)
    num_cols = len(imgs_ls[0]) + with_orig
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        figsize=(img_size, img_size),
    )
    for row_idx, row in enumerate(imgs_ls):
        row = [orig_img] + row if with_orig else row  # noqa: RUF005
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if col_title is not None:
        for ax, title in zip(axs.flatten(), col_title):
            ax.set_title(f"{title}", fontsize=font_s)
    else:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(font_s)

    plt.tight_layout()


############################
#### Metrics functions ####
###########################


def psnr(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float | Any:
    """
    psnr

    Compute the 'Peak Signal-to-Noise Ratio' (psnr) metric
    from two given images.
    High PSNR Value: Indicates high image quality.
    NOTE: the two images must have the same dimension

    ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Parameters
    ----------
    img1 : npt.NDArray[np.uint8]
        reference image
    img2 : npt.NDArray[np.uint8]
        reconstructed image

    Returns
    -------
    float | Any
        the psnr value from the two images
    """
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(
    imageA: npt.NDArray[np.uint8],
    imageB: npt.NDArray[np.uint8],
    nband: int,
) -> float | Any:
    """
    mse

    Compute the 'Mean Squared Error' (mse) metric
    from two given images. The mse is the
        sum of the squared difference between the two images.
    The lower the value the more similar.
    NOTE: the two images must have the same dimension


    Parameters
    ----------
    imageA : npt.NDArray[np.uint8]
        reference image
    imageB : npt.NDArray[np.uint8]
        reconstructed image
    nband : int
        number of channels in the image? -> need to check!

    Returns
    -------
    float | Any
        the mse value from the two images
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * nband)

    return err


def mae(
    imageA: npt.NDArray[np.uint8],
    imageB: npt.NDArray[np.uint8],
    nband: int,
) -> float | Any:
    """
    mae

    Compute the 'Mean Absolute Error' (mae) metric
    from two given images. The mse between the two images is the
        sum of the squared difference between the two images.
    NOTE: the two images must have the same dimension

    Parameters
    ----------
    imageA : npt.NDArray[np.uint8]
        reference image
    imageB : npt.NDArray[np.uint8]
        reconstructed image
    nband : int
        number of channels in the image? -> need to check!

    Returns
    -------
    float | Any
        _description_
    """
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * nband)
    return err


def rmse(
    imageA: npt.NDArray[np.uint8],
    imageB: npt.NDArray[np.uint8],
    nband: int,
) -> float | Any:
    """
    rmse

    Compute the 'Root Mean Squared Error' (rmse) metric
    from two given images.
    The rmse between the two images is the
        sum of the squared difference between the two images.
    NOTE: the two images must have the same dimension


    Parameters
    ----------
    imageA : npt.NDArray[np.uint8]
        reference image
    imageB : npt.NDArray[np.uint8]
        reconstructed image
    nband : int
        number of channels in the image? -> need to check!

    Returns
    -------
    float | Any
        the rmse value from the two images
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * nband)
    return np.sqrt(err)
