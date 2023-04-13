import random
import numpy as np
import math
import scipy
import torch.nn.functional as F


def generate_perturbation_matrix_3D(max_t=5, max_s=1.4, min_s=0.7, max_r=5):
    ## Very primitive data augmentation. Need better augmentation.
    # Translation
    tx = random.randint(-max_t, max_t + 1)
    ty = random.randint(-max_t, max_t + 1)
    tz = random.randint(-max_t, max_t + 1)
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    # Scaling
    sx = np.random.uniform(min_s, max_s)
    sy = np.random.uniform(min_s, max_s)
    sz = np.random.uniform(min_s, max_s)
    S = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
    # Rotation
    rx = random.randint(-max_r, max_r + 1)  # x-roll (w.r.t x-axis)
    ry = random.randint(-max_r, max_r + 1)  # y-roll
    rz = random.randint(-max_r, max_r + 1)  # z-roll
    c = math.cos(math.pi * rx / 180)
    s = math.sin(math.pi * rx / 180)
    Rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    c = math.cos(math.pi * ry / 180)
    s = math.sin(math.pi * ry / 180)
    Ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    c = math.cos(math.pi * rz / 180)
    s = math.sin(math.pi * rz / 180)
    Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.dot(np.dot(Ry, Rz), Rx)
    # R = np.eye(4)
    # Generate perturbation matrix
    G = np.dot(np.dot(S, T), R)
    # G = np.dot(S,R)
    return G


def augment_im(im, affine_mat, crops=None):
    if crops is not None:
        h_lo = crops[0]
        h_hi = crops[1]
        im = im[h_lo:h_hi, ...]

    dims = len(im.shape)

    G_mat = affine_mat[:dims, :dims]
    G_offset = affine_mat[:dims, dims]
    im_aug = scipy.ndimage.affine_transform(im, G_mat, offset=G_offset, order=1)
    return im_aug


def augment_cents(cents, im_shape, affine_mat, crops=None):
    if crops is not None:
        h_lo = crops[0]
        h_hi = crops[1]
        cents[cents[:, 0] < h_lo, :] = np.nan
        cents[cents[:, 0] > h_hi, :] = np.nan
        cents[:, 0] -= h_lo

    dims = len(im_shape)

    G_mat = affine_mat[:dims, :dims]
    G_offset = affine_mat[:dims, dims]

    cents_aug = np.dot(np.linalg.inv(G_mat), (cents - G_offset).transpose(1, 0)).transpose(1, 0)

    return cents_aug


def pad_to_shape(arr, out_shape):
    """ignore last is TRUE when you are trying to pad a image with channels
    """
    dims = len(arr.shape)

    pads = ()
    for dim in range(dims):

        old_len = arr.shape[dim]
        new_len = out_shape[dim]

        diff = new_len - old_len

        if diff % 2 == 0:
            pads += (diff // 2, diff // 2)
        else:
            pads += (diff // 2, diff // 2 + 1)

    # pytorch for some reason take pads from back!
    new_arr = F.pad(arr, pads[::-1])

    return new_arr, pads  # pads is a tuple. every two elems corr. to top and bottom pads for each dim.


def centroids_to_mask(cents, im_shape, sigma):
    h, w, d = im_shape
    chs = np.shape(cents)[0]

    dh, dw, dd = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing='ij')

    d_hwd = np.stack((dh, dw, dd), 3)  # h*w*d*3

    d_hwd = np.expand_dims(d_hwd, -2)  # h*w*d*1*3
    d_hwd = np.tile(d_hwd, [1, 1, 1, chs, 1])  # h*w*d*chs*3

    cents = cents[np.newaxis, np.newaxis, np.newaxis, :, :]  # 1*1*1*chs*3

    squared_diff = np.square(d_hwd.astype(float) - cents)

    distance_map = np.sum(squared_diff, -1)

    heatmap = np.exp(- distance_map / (2 * (sigma ** 2)))

    # heatmap = (1/(sigma * math.sqrt(2 * math.pi)))*heatmap

    heatmap = np.where(np.isnan(heatmap), np.zeros_like(heatmap), heatmap)

    # normalise center to one

    # heatmap = rescale_tf(heatmap)

    return heatmap