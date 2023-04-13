import unittest

import torch

from hr_net_3d import HighResolutionNet


class TestHrNet3D(unittest.TestCase):

    def test_init(self):

        hr_net = HighResolutionNet(out_chs=2)

        x = torch.zeros((1, 1, 50, 50, 50))
        y, feat = hr_net.forward(x)
        self.assertEqual(y.shape, (1, 2, 13, 13, 13))

