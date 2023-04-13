import unittest
from hr_net_3d import Runner


class TestRunner(unittest.TestCase):

    def setUp(self):

        labels = ["Label A", "Label B", "Label C"]

        params = {
            'name': "TestRunner",
            'gpu': -1,
            'seed': 1,
            'data_dir': "",
            'log_dir': "",
            'save_dir': "",
            'labels': labels,
            'lr': 0.001,
            'l2': 0.0,
            'max_epochs': 400
        }

        data_iter = {}

        self.runner = Runner(data_iter, params)

    def test_init(self):

        self.assertIsNotNone(self.runner)

