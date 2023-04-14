import unittest
import numpy as np
import torch
from hr_net_3d import Runner
from data_manager import get_data_iter, CSI_LABELS, VERSE_LABELS
import matplotlib.pyplot as plt


params = {
    'name': "TestRunner",
    'gpu': -1,
    'seed': 1,
    'data_dir': "",
    'log_dir': "",
    'save_dir': "",
    'lr': 0.001,
    'l2': 0.0,
    'max_epochs': 1
}


class TestHrNet3D(unittest.TestCase):

    def test_fit(self):

        data_iter = get_data_iter("data/verse2019", "train", "test")
        params["labels"] = VERSE_LABELS

        runner = Runner(data_iter, params)
        print("training...")
        runner.fit()

    def test_load(self):
        params["labels"] = CSI_LABELS
        runner = Runner({}, params)
        runner.load_model("models/hrnet_3d_csi_2mm")

    def test_prediction(self):
        params["labels"] = CSI_LABELS
        runner = Runner({}, params)
        runner.load_model("models/hrnet_3d_csi_2mm")

        img_id = 4517454
        img_data = np.load(f"data/csi/eval/{img_id}_2mm_3d.npz")

        im = img_data["im"]
        # -- normalise
        im = im / 2048.
        im[im < -1] = -1
        im[im > 1] = 1

        prediction = runner.predict(torch.unsqueeze(torch.unsqueeze(torch.Tensor(im), 0), 0))

        print(im.shape)
        print(prediction.shape)

        heatmap = np.sum(prediction.detach().cpu().numpy(), axis=-1)

        fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
        ax[0, 0].imshow(np.max(im, axis=2), cmap="gray")
        ax[1, 0].imshow(np.max(im, axis=1), cmap="gray")
        ax[0, 1].imshow(np.max(heatmap, axis=2))
        ax[1, 1].imshow(np.max(heatmap, axis=1))

        plt.show()

