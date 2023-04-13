import pickle
import os
print(os.getcwd())
from gaussian_crf import GaussianCRF, PlotGaussianCRF
import matplotlib.pyplot as plt
import networkx as nx


def main():

    print("""
This example demonstrates the interactive gaussian crf 
for vertebrae localization in spine CT images on image 
'070' of the VerSe 2019 benchmark dataset [1].
    
[1] Anjany Sekuboyina, Malek E Husseini, Amirhossein 
Bayat, Maximilian Loeffler, Hans Liebl, et al., "Verse: 
A vertebrae labelling and segmentation benchmark for 
multi-detector ct images," MedIA, vol. 73, pp. 102166, 2021.
    """)

    with open("spine_example_data.pickle", "rb") as handle:
        data = pickle.load(handle)

    # create gaussian crf
    spine = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
             'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
             'L1', 'L2', 'L3', 'L4', 'L5']
    graph = nx.Graph([(spine[i], spine[i+1]) for i in range(len(spine)-1)])
    crf = GaussianCRF(graph, dim=3)

    # set the unary and pairwise distributions from the loaded data
    for v in graph.nodes:
        crf.set_mean(v, data["unary_means"][v])
        crf.set_precision(v, data["unary_precisions"][v])
    for e in graph.edges:
        crf.set_mean(e, data["pairwise_means"][e])
        crf.set_precision(e, data["pairwise_precisions"][e])

    # create a figure with two axis, one for the sagittal projection, one for the coronal projection
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
    for a in ax:
        a.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.02)

    # show the sagittal and coronal projection of ct image of the spine
    img_s = data["img_sagittal"]
    ax[0].imshow(img_s, cmap="gray", extent=(-0.5, img_s.shape[1]/2 - 0.5, img_s.shape[0]/2 - 0.5, -0.5))
    ax[0].set_title("Sagittal projection")
    img_c = data["img_coronal"]
    ax[1].imshow(img_c, cmap="gray", extent=(-0.5, img_c.shape[1]/2 - 0.5, img_c.shape[0]/2 - 0.5, -0.5))
    ax[1].set_title("Coronal projection")

    # plot the crf
    plotter = PlotGaussianCRF(crf, ax, projections=[[1, 0], [2, 0]], label=True,
                              marker_size=40, unary_label=True, font_size=20)
    plotter.plot()

    # set the ax limits to the extent of the images
    ax[0].set_xlim(-0.5, img_s.shape[1]/2 - 0.5)
    ax[1].set_xlim(-0.5, img_c.shape[1] / 2 - 0.5)
    ax[0].set_ylim(img_s.shape[0]/2 - 0.5, -0.5)

    # show the figure
    plt.show()


if __name__ == "__main__":
    main()
