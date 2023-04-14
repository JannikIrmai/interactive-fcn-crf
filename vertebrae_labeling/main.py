import matplotlib.pyplot as plt
import numpy as np
from hr_net_3d import Runner
from data_manager import CSI_LABELS
from interactive_gaussian_crf import GaussianCRF, PlotGaussianCRF
import networkx as nx
import pickle
import torch


def fit_unary_from_masks(msk: np.ndarray):
    n = msk.ndim - 1
    mesh = np.meshgrid(*[np.linspace(0, s - 1, s, dtype=np.float32) for s in msk.shape[1:]], indexing='ij')
    mesh = np.array([a.flatten() for a in mesh])

    mean = np.zeros((msk.shape[0], n))
    cov = np.zeros((msk.shape[0], n, n))

    for i in range(msk.shape[0]):
        weights = msk[i].flatten()
        if weights.sum() == 0:
            mean[i] = np.array(msk.shape[1:]) / 2
        else:
            mean[i] = np.average(mesh, axis=1, weights=weights)
            cov[i] = np.cov(mesh, aweights=weights, ddof=0)
    return mean, cov


def interactive_fcn_crf(runner: Runner, crf: GaussianCRF, img: np.ndarray):

    heatmap = runner.predict(torch.unsqueeze(torch.unsqueeze(torch.Tensor(img), 0), 0)).detach().cpu().numpy()
    # move channel axis to front
    heatmap = np.moveaxis(heatmap, -1, 0)

    t_cov = 0.3
    t_noise = 0.1

    # remove noise from the heatmap
    heatmap[heatmap < t_noise] = 0
    # compute the maximal heatmap value for each channel
    max_val = np.max(heatmap, axis=(1, 2, 3))
    # compute the means and covariances of the channels
    mean, cov = fit_unary_from_masks(heatmap)

    # set the unaries of the crf
    for v in crf.get_variables():
        i = runner.p["labels"].index(v)
        crf.set_mean(v, mean[i])
        if max_val[i] > t_cov:
            c = cov[i] / max_val[i]
            p = np.linalg.inv(c)
            crf.set_precision(v, p)
        else:
            crf.set_precision(v, np.eye(3) * 1e-6)

    # create figure
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))

    # plot maximum intensity projections of image
    im_s, im_c = np.amax(img, -1), np.amax(img, -2)
    ax[0].imshow(im_s, cmap="gray", extent=(-0.5, heatmap.shape[2] - 0.5, heatmap.shape[1] - 0.5, -0.5))
    ax[1].imshow(im_c, cmap="gray", extent=(-0.5, heatmap.shape[3] - 0.5, heatmap.shape[1] - 0.5, -0.5))
    ax[0].set_title("Sagittal projection")
    ax[1].set_title("Coronal projection")

    # plot crf
    plotter = PlotGaussianCRF(crf, ax, projections=[[1, 0], [2, 0]], label=True,
                              marker_size=40, unary_label=True, font_size=20)
    plotter.plot()

    # set the ax limits to the extent of the images
    ax[0].set_xlim(-0.5, heatmap.shape[2] - 0.5)
    ax[0].set_ylim(heatmap.shape[1] - 0.5, -0.5)
    ax[1].set_xlim(-0.5, heatmap.shape[3] - 0.5)

    plt.show()


def main():

    # load the FCN
    params = {
        'name': "CSIRunner",
        'gpu': -1,
        'seed': 1,
        'data_dir': "",
        'log_dir': "",
        'save_dir': "",
        'lr': 0.001,
        'l2': 0.0,
        'max_epochs': 1,
        'labels': CSI_LABELS
    }
    runner = Runner({}, params)
    runner.load_model("models/hrnet_3d_csi_2mm")

    # load the CRF
    graph = nx.Graph([(CSI_LABELS[i], CSI_LABELS[i + 1]) for i in range(len(CSI_LABELS) - 1)])
    crf = GaussianCRF(graph, dim=3)
    with open("models/csi_spine_prior.pickle", "rb") as handle:
        data = pickle.load(handle)
    for e in graph.edges:
        crf.set_mean(e, data["pairwise_means"][e])
        crf.set_precision(e, data["pairwise_precisions"][e])

    # load a test image
    img_id = 4517454
    img_data = np.load(f"data/csi/eval/{img_id}_2mm_3d.npz")
    im = img_data["im"]
    # -- normalise
    im = im / 2048.
    im[im < -1] = -1
    im[im > 1] = 1

    interactive_fcn_crf(runner, crf, im)


if __name__ == "__main__":
    main()
