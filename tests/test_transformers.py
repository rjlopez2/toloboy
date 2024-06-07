from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image
from toloboy import toloboy as to

url = "https://perritos.myasustor.com:1985/data/img_id_1550.jpg"
response = requests.get(url)  # noqa: S113
img = Image.open(BytesIO(response.content))


def test_from_LAB_to_RGB_img(img_test: Any = img) -> None:
    # retrieve a sample image
    img_test_np = np.asarray(img_test)
    x_s, y_s, _ = img_test_np.shape
    r, g, b = (img_test_np[:, :, i].reshape((x_s * y_s, 1)) for i in range(3))
    l, a, b = to.RGB2LAB(r, g.reshape(-1, 1), b.reshape(-1, 1))  # noqa: E741
    ab = np.concatenate((a, b), axis=1).reshape((x_s, y_s, -1))
    new_img = to.from_LAB_to_RGB_img(l, ab)
    assert np.allclose(img_test_np, new_img.reshape((x_s, y_s, -1)))


def test_RGB2LAB() -> None:
    # this is just a dumb test.
    # No sure what/how to test here
    # may be check if the results are comparable to opencv lib?

    # img_test_np = np.asarray(img)
    # x_s, y_s, _ = img_test_np.shape
    # r, g, b  = (img_test_np[:,:,i].reshape((x_s*y_s, 1)) for i in range(3))
    # _ = to.RGB2LAB(r, g.reshape(-1, 1), b.reshape(-1, 1) )
    assert True


def test_LAB22RGB() -> None:
    # this is just a dumb test.
    # No sure what/how to test here

    # img_test_np = np.asarray(img)
    # x_s, y_s, _ = img_test_np.shape
    # r, g, b  = (img_test_np[:,:,i].reshape((x_s*y_s, 1)) for i in range(3))
    # l, a, b = to.RGB2LAB(r, g.reshape(-1, 1), b.reshape(-1, 1) )
    # r_r, g_r, b_r= to.LAB22RGB(r, g.reshape(-1, 1), b.reshape(-1, 1) )

    # assert (r, g, b ) == (r_r, g_r, b_r)
    assert True


def test_plot_multiple_imgs_test() -> None:
    # this is just a dumb test.

    assert True


def test_psnr_test(img_test: Any = img) -> None:
    # this is just a dumb test.
    img_test_np = np.asarray(img_test)
    rms_img_test = to.psnr(imageA=img_test_np, imageB=img_test_np)

    assert np.allclose(rms_img_test, rms_img_test)


def test_mse_test(img_test: Any = img) -> None:
    # this is just a dumb test.
    img_test_np = np.asarray(img_test)
    rms_img_test = to.mse(imageA=img_test_np, imageB=img_test_np, nband=2)

    assert np.allclose(rms_img_test, rms_img_test)


def test_mae_test(img_test: Any = img) -> None:
    # this is just a dumb test.
    img_test_np = np.asarray(img_test)
    rms_img_test = to.mae(imageA=img_test_np, imageB=img_test_np, nband=2)

    assert np.allclose(rms_img_test, rms_img_test)


def test_rmse_test(img_test: Any = img) -> None:
    # this is just a dumb test.
    img_test_np = np.asarray(img_test)
    rms_img_test = to.rmse(imageA=img_test_np, imageB=img_test_np, nband=2)

    assert np.allclose(rms_img_test, rms_img_test)
