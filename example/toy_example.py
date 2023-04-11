from gaussian_crf import GaussianCRF, PlotGaussianCRF
import networkx as nx
import matplotlib.pyplot as plt


def main():
    # create a graph with three nodes 'a', 'b', and 'c'
    g = nx.Graph()
    g.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])

    # create a Gaussian CRF based on this graph where each node corresponds to a 2-dimensional gaussian random variable
    crf = GaussianCRF(g, dim=2)

    # set the mean vectors of the unary distributions associated with the three variables
    crf.set_mean('a', [1, 0])
    crf.set_mean('b', [-1, 0])
    crf.set_mean('c', [0, -1])

    # set the mean vectors of the pairwise distributions associated with all edges in the graph
    crf.set_mean(('a', 'b'), [2, 0])  # mean vector of location of a minus location of b
    crf.set_mean(('a', 'c'), [1, 1])
    crf.set_mean(('b', 'c'), [-1, 1])

    # The covariances/precisions of the unary and pairwise distributions are not specified in this example.
    # The defaults are unary covariance/precision. The precision matrix for a given unary or pairwise distribution,
    # can be set with the set_precision() method.

    # create an interactive plot of the gaussian crf
    p = PlotGaussianCRF(crf, label=True, unary_label=True, marker_size=100, font_size="xx-large")
    p.plot()
    plt.show()


if __name__ == "__main__":
    main()
