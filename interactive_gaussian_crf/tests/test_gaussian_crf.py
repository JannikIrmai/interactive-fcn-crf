from gaussian_crf import GaussianCRF, HigherOrderGaussianCRF, PlotGaussianCRF
import unittest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class TestGaussianCRF(unittest.TestCase):

    def test_infer_2d_toy(self):

        crf = GaussianCRF(n=2, dim=2)
        mean, cov = crf.infer()
        self.assertTrue(np.allclose(np.array(list(mean.values())), np.zeros((2, 2))))
        # self.assertTrue(np.allclose(precision, [[[[2, 0], [0, 2]], [[-1, 0], [0, -1]]],
        #                                         [[[-1, 0], [0, -1]], [[2, 0], [0, 2]]]]))

        crf._unary_means[1] = np.array([2, 1])
        crf._pairwise_means[0, 1] = np.array([-1, -1])
        mean, cov = crf.infer()
        self.assertTrue(np.allclose(np.array(list(mean.values())), [[1/3, 0], [5/3, 1]]))
        # self.assertTrue(np.allclose(precision, [[[[2, 0], [0, 2]], [[-1, 0], [0, -1]]],
        #                                         [[[-1, 0], [0, -1]], [[2, 0], [0, 2]]]]))

        mean, cov = crf.infer_with_evidence({1: np.array([2, 2])})
        self.assertTrue(np.allclose(np.array(list(mean.values())), [[1/2, 1/2], [2, 2]]))
        # self.assertTrue(np.allclose(precision, [[[[2, 0], [0, 2]], [[0, 0], [0, 0]]],
        #                                         [[[0, 0], [0, 0]], [[np.infty, 0], [0, np.infty]]]]))

        mean, cov = crf.infer_with_evidence({0: np.array([1, 0])})
        self.assertTrue(np.allclose(np.array(list(mean.values())), [[2, 1], [1, 0]]))
        # self.assertTrue(np.allclose(precision, [[[[np.infty, 0], [0, np.infty]], [[0, 0], [0, 0]]],
        #                                         [[[0, 0], [0, 0]], [[2, 0], [0, 2]]]]))

    def test_zero_precision(self):
        crf = GaussianCRF(n=3, dim=2)
        mean, cov = crf.infer()
        self.assertTrue(np.allclose(np.array(list(mean.values())), np.zeros((3, 2))))

        crf._unary_means[0] = np.array([0, 0])
        crf._unary_means[1] = np.array([2, 0])
        crf._unary_precisions[2] = np.zeros((2, 2))
        crf._pairwise_means[0, 1] = np.array([-2, 0])
        crf._pairwise_means[0, 2] = np.array([-1, -3])
        crf._pairwise_means[1, 2] = np.array([1, -3])
        mean, cov = crf.infer()
        self.assertTrue(np.allclose(np.array(list(mean.values())), [[0, 0], [2, 0], [1, 3]]))
        # self.assertTrue(np.allclose(precision, [[[[3,  0], [0,  3]], [[-1,  0], [0, -1]], [[-1,  0], [0, -1]]],
        #                                         [[[-1,  0], [0, -1]], [[3,  0], [0,  3]], [[-1,  0], [0, -1]]],
        #                                         [[[-1,  0], [0, -1]], [[-1,  0], [0, -1]], [[2,  0], [0,  2]]]]))


class TestHigherOrderGaussianCRF(unittest.TestCase):

    def test_higher_order_crf(self):

        crf = HigherOrderGaussianCRF(3, [((0,), (1,)), ((0, 1), (1, -1)), ((0, 1, 2), (1, -2, 1))], 2)
        crf.set_mean(1, np.array([-2, 0]))

        mean, cov = crf.infer_with_evidence({0: np.array([1, 1])})
        self.assertTrue(np.allclose(mean[0], [1, 1]))
        self.assertTrue(np.allclose(mean[1], [3, 1]))
        self.assertTrue(np.allclose(mean[2], [5, 1]))

    def test_log_likelihood(self):

        crf = HigherOrderGaussianCRF(1, [((0,), (1,))], 2)
        crf.set_precision(0, np.array([[1, 0], [0, 0.5]]))
        ll = crf.log_likelihood({0: np.array([0, 0])})
        self.assertAlmostEqual(np.log(1 / (2*np.pi * np.sqrt(2))), ll)

        ll = crf.log_likelihood({0: np.array([1, 2])})
        self.assertAlmostEqual(np.log(1 / (2*np.pi * np.sqrt(2))) - 3/2, ll)

    def test_ll_missing_unary(self):

        n = 10
        edges = [((i,), (1,)) for i in range(n)]
        edges += [((i, i+1), (1, -1)) for i in range(n-1)]
        crf = HigherOrderGaussianCRF(n, edges, 2)
        for i in range(n):
            crf.set_mean(i, np.array([i, 0]))
        for i in range(n-1):
            crf.set_mean(n+i, np.array([-1, 0]))
        ll_with_unary = crf.log_likelihood(crf.infer()[0])
        for i in range(1, n):
            crf.set_precision(i, np.zeros((2, 2)))
        ll_without_unary = crf.log_likelihood(crf.infer()[0])
        self.assertLess(ll_without_unary, ll_with_unary)


class TestPlotGaussianCRF(unittest.TestCase):

    def test_with_names(self):
        g = nx.Graph()
        g.add_edges_from([('a', 'b'), ('b', 'c')])
        crf = GaussianCRF(g, dim=2)
        for e in g.edges:
            crf.set_mean(e, np.ones(2))

        p = PlotGaussianCRF(crf, label=True, unary_label=True, marker_size=100, font_size="xx-large")
        p.plot()
        plt.show()
        print(p.get_centroids())

    def test_plot_path_graph(self):
        n = 5
        d = 5
        crf = GaussianCRF(nx.path_graph(n), n=5, dim=2)
        for i in range(n):
            crf._unary_means[i] = np.array([i*d, i*d])
        for i in range(n-1):
            crf._pairwise_means[i, i+1] = np.array([-d, -d])

        p = PlotGaussianCRF(crf, label=True)
        p.plot()
        plt.show()

    def test_plot_3d_gaussian(self):
        crf = GaussianCRF(n=2, dim=3)
        crf._unary_means[1] = np.array([2, 3, 4])
        cov = [[2, 0, 1], [0, 1, 0], [1, 0, 4]]
        crf._unary_precisions[1] = np.linalg.inv(cov)
        crf._pairwise_means[0, 1] = np.array([-3, -3, -3])
        cov = [[1, 0, 0], [0, 2, 1], [0, 1, 3]]
        crf._pairwise_precisions[0, 1] = np.linalg.inv(cov)

        fig = plt.figure()
        ax01 = fig.add_subplot(2, 2, 1)
        ax02 = fig.add_subplot(2, 2, 3, sharex=ax01)
        ax12 = fig.add_subplot(2, 2, 4, sharey=ax02)

        p = PlotGaussianCRF(crf, ax=[ax01, ax02, ax12], projections=[[0, 1], [0, 2], [1, 2]],
                            label=True)
        for ax in p._ax:
            ax.set_aspect('equal')
        p.plot()
        plt.show()

    def test_plot_single_2d_gaussian(self):
        crf = HigherOrderGaussianCRF(1, [((0,), (1,))], 2)
        crf.set_precision(0, np.array([[1, 0], [0, 1/4]]))
        crf_plt = PlotGaussianCRF(crf)
        crf_plt.plot()
        plt.show()

    def test_plot_higher_order_crf(self):
        n = 10
        d = 5

        edges = [((0,), (1,)),
                 ((n-1,), (1,))]
        for i in range(n-2):
            edges.append(((i, i+1, i+2), (1, -2, 1)))

        crf = HigherOrderGaussianCRF(n, edges, 2)
        crf.set_mean(1, np.array([(n-1)*d, 0]))
        for i in range(n-2):
            crf.set_mean(i+2, np.array([0, 0]))

        crf_plt = PlotGaussianCRF(crf)
        crf_plt._ax[0].set_aspect('equal')
        crf_plt.plot()
        plt.show()

    def test_degenerate_model(self):
        n = 10
        edges = [((i,), (1,)) for i in range(n)]
        edges += [((i, i + 1), (-1, 1)) for i in range(n-1)]
        crf = HigherOrderGaussianCRF(n, edges, 2)
        for i in range(n):
            crf.set_precision(i, 0.05 * np.eye(2))
        for i in range(n-1):
            crf.set_mean(n+i, np.array([0, 5]))
        crf_plt = PlotGaussianCRF(crf)
        crf_plt._stds = [1]
        crf_plt._ax[0].set_aspect('equal')
        crf_plt.plot()
        plt.show()


if __name__ == "__main__":
    unittest.main()
