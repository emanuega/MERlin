import cv2
import numpy as np
import random

import merlin.util.deconvolve as deconvolve
import merlin.util.matlab as matlab


decon_sigma = 2
decon_filter_size = 9


def decon_diff(image, gt_image):
    on_gt = np.sum(image[(gt_image > 0)])
    off_gt = np.sum(image[gt_image == 0])

    return (on_gt/(on_gt + off_gt))


def make_image():
    # Always make the same image.
    random.seed(42)

    # Ground truth.
    gt_image = np.zeros((100, 150))
    for i in range(40):
        x = random.randint(5, 95)
        y = random.randint(5, 145)
        gt_image[x, y] = random.randint(10, 50)

    [pf, pb] = deconvolve.calculate_projectors(64, decon_sigma)
    image = cv2.filter2D(gt_image, -1, pf, borderType=cv2.BORDER_REPLICATE)

    return [image, gt_image]


def test_deconvolve_lucyrichardson():
    [image, gt_image] = make_image()

    d1 = decon_diff(image, gt_image)
    d_image = deconvolve.deconvolve_lucyrichardson(image,
                                                   decon_filter_size,
                                                   decon_sigma,
                                                   20)
    d2 = decon_diff(d_image, gt_image)
    print(d1, d2)

    assert (d2 > d1)


def test_deconvolve_lucyrichardson_guo():
    [image, gt_image] = make_image()

    d1 = decon_diff(image, gt_image)
    d_image = deconvolve.deconvolve_lucyrichardson_guo(image,
                                                       decon_filter_size,
                                                       decon_sigma,
                                                       2)
    d2 = decon_diff(d_image, gt_image)
    print(d1, d2)

    assert (d2 > d1)
