import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backend_bases import MouseButton, MouseEvent, KeyEvent
from .models import CRFBase
import numpy as np


def covariance_to_height_width_angle(cov):
    ((a, b), (b, c)) = cov
    tmp = np.sqrt((a - c) ** 2 / 4 + b ** 2)
    lambda_1 = (a + c) / 2 + tmp
    lambda_2 = (a + c) / 2 - tmp
    if b == 0:
        angle = 0 if a >= c else np.pi / 2
    else:
        angle = np.arctan2(lambda_1 - a, b)
    return 2 * np.sqrt(lambda_1), 2 * np.sqrt(lambda_2), angle / np.pi * 180


class PlotGaussianCRF:

    def __init__(self, crf: CRFBase, ax: (list, plt.Axes) = None, colors: dict = None, projections: list = None,
                 label: bool = False, unary_label: bool = False,
                 marker_size: float = None, map_marker: str = "o", unary_marker: str = "+",
                 font_size: (float, str) = None):

        self._crf = crf
        self._evidence = dict()

        # if projections are not specified set to default, otherwise assert that projections are valid
        if projections is None:
            self._projections = [[0, 1]]
        else:
            for p in projections:
                if len(p) != 2 or not isinstance(p, list) or p[0] == p[1] or \
                        p[0] not in range(self._crf.get_dim()) or p[1] not in range(self._crf.get_dim()):
                    raise ValueError(f"Invalid projection {p}")
            self._projections = projections

        if ax is None:
            self._fig, self._ax = plt.subplots(1, len(self._projections))
            if len(self._projections) == 1:
                self._ax = [self._ax]
            else:
                self._ax = list(self._ax)
        else:
            if isinstance(ax, plt.Axes):
                self._fig = ax.get_figure()
                self._ax = [ax]
            else:
                self._fig = ax[0].get_figure()
                self._ax = [a for a in ax]

        self._label = label
        self._unary_label = unary_label

        # functions for making the plot interactive
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self._fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # colors of the different variables
        if colors is None:
            cm = plt.get_cmap('tab10')
            self._colors = {v: cm(i % 10) for i, v in enumerate(crf.get_variables())}
        else:
            self._colors = colors

        self.marker_size = marker_size
        self.map_marker = map_marker
        self.unary_marker = unary_marker
        self.font_size = font_size

        # means of unary distributions
        self._centroids, _ = self._crf.infer()

        # attributes for tracking a variable that is moved
        self._selected_centroid = None
        self._moved = False

        # attributes for maintaining the matplotlib objects that are drawn on the axis
        self._map_scatter = dict()
        self._map_ellipsis = {i: dict() for i in range(len(self._projections))}
        self._unary_plots = []
        self._label_handles = {i: dict() for i in range(len(self._projections))}

        # parameters for plotting
        self._stds = [1, 2, 3]
        self._rel_dist = 0.05

    def remove(self):
        for handle in self._map_scatter.values():
            handle.remove()
        self._map_scatter.clear()
        for ell in self._map_ellipsis.values():
            for e in ell.values():
                e.remove()
            ell.clear()
        for obj in self._unary_plots:
            obj.remove()
        self._unary_plots.clear()
        for labels in self._label_handles.values():
            for handle in labels.values():
                handle.remove()
            labels.clear()

    def plot(self):
        self._plot_map()

    def plot_unary(self):
        for obj in self._unary_plots:
            obj.remove()
        self._unary_plots = []

        for i, p in enumerate(self._projections):

            for v in self._crf.get_variables():
                mean = self._crf.get_mean(v)
                self._unary_plots.append(self._ax[i].scatter([mean[p[0]]], [mean[p[1]]], marker=self.unary_marker,
                                                             color=self._colors[v], s=self.marker_size))
                self._unary_plots.append(self._ax[i].text(mean[p[0]], mean[p[1]], s="  " + str(v),
                                                          color=self._colors[v], size=self.font_size))
                try:
                    cov = np.linalg.inv(self._crf.get_precision(v))
                    cov = cov[p][:, p]
                    height, width, angle = covariance_to_height_width_angle(cov)
                    for std in self._stds:
                        e = Ellipse(mean[p], std * height, std * width, angle=angle, fc=[0] * 4, ec=self._colors[v],
                                    linestyle=":")
                        self._unary_plots.append(self._ax[i].add_patch(e))
                except np.linalg.LinAlgError:
                    pass

    def _plot_map(self):
        mean_dict, cov_dict = self._crf.infer_with_evidence(self._evidence)
        self._centroids = mean_dict

        for i, p in enumerate(self._projections):
            # scatter map centers
            if i not in self._map_scatter:
                self._map_scatter[i] = self._ax[i].scatter([mean_dict[v][p[0]] for v in self._crf.get_variables()],
                                                           [mean_dict[v][p[1]] for v in self._crf.get_variables()],
                                                           color=[self._colors[v] for v in self._crf.get_variables()],
                                                           s=self.marker_size, marker=self.map_marker)
            else:
                self._map_scatter[i].set_offsets([mean_dict[v][p] for v in self._crf.get_variables()])
            ec = ['black' if v in self._evidence else [0] * 4 for v in self._crf.get_variables()]
            self._map_scatter[i].set_ec(ec)

            for v in self._crf.get_variables():
                # write labels
                if self._label:
                    if v in self._label_handles[i]:
                        text = self._label_handles[i][v]
                        text.set_position(mean_dict[v][p])
                    else:
                        text = self._ax[i].text(*mean_dict[v][p], s="  " + str(v), size=self.font_size,
                                                color=self._colors[v])
                        self._label_handles[i][v] = text
                # update ellipsis
                if v in self._evidence:
                    if v in self._map_ellipsis[i]:
                        e = self._map_ellipsis[i].pop(v)
                        e.remove()
                    continue
                cov = cov_dict[v][p][:, p]
                height, width, angle = covariance_to_height_width_angle(cov)
                if v in self._map_ellipsis[i]:
                    e = self._map_ellipsis[i][v]
                    e.height = height
                    e.width = width
                    e.angle = angle
                    e.center = mean_dict[v][p]
                else:
                    e = Ellipse(mean_dict[v][p], height, width, angle=angle, fc=[0] * 4, edgecolor=self._colors[v])
                    self._map_ellipsis[i][v] = e
                    self._ax[i].add_patch(e)

    def _on_key_press(self, event: KeyEvent):
        print("pressed", event.key)
        if event.key == 'u':
            if len(self._unary_plots) == 0:
                self.plot_unary()
            else:
                for obj in self._unary_plots:
                    obj.remove()
                self._unary_plots = []
            self._fig.canvas.draw_idle()
        if event.key == "m":
            for scatter in self._map_scatter.values():
                scatter.set_visible(not scatter.get_visible())
            for ell in self._map_ellipsis.values():
                for e in ell.values():
                    e.set_visible(not e.get_visible())
            for labels in self._label_handles.values():
                for label in labels.values():
                    label.set_visible(not label.get_visible())
            self._fig.canvas.draw_idle()
        if event.key == "e":
            for ell in self._map_ellipsis.values():
                for e in ell.values():
                    e.set_visible(not e.get_visible())
            self._fig.canvas.draw_idle()

    def _on_click(self, event: MouseEvent):
        if event.button != MouseButton.LEFT:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._fig.canvas.manager.toolbar.mode.name != "NONE":
            return

        # check if the axes is in self._ax
        if event.inaxes not in self._ax:
            return
        i = self._ax.index(event.inaxes)

        x_lim = self._ax[i].get_xlim()
        y_lim = self._ax[i].get_ylim()
        size = min(abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]))

        min_dist = np.inf
        for v, cent in self._centroids.items():
            diff = cent[self._projections[i]] - [event.xdata, event.ydata]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < min_dist:
                self._selected_centroid = v
                min_dist = dist
        if min_dist > size * self._rel_dist:
            print("Missed!")
            self._selected_centroid = False
        self._moved = False

    def _on_move(self, event: MouseEvent):
        if self._selected_centroid is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in self._ax:
            return
        i = self._ax.index(event.inaxes)

        self._moved = True
        evidence = self._centroids[self._selected_centroid]
        evidence[self._projections[i]] = [event.xdata, event.ydata]
        self._evidence[self._selected_centroid] = evidence
        self._plot_map()
        self._fig.canvas.draw_idle()

    def _on_release(self, event: MouseEvent):
        if event.button != MouseButton.LEFT:
            return
        if self._selected_centroid is None:
            return
        if not self._moved and self._selected_centroid in self._evidence:
            self._evidence.pop(self._selected_centroid)
            self._plot_map()
            self._fig.canvas.draw_idle()
        self._selected_centroid = None

    def get_centroids(self):
        return self._centroids.copy()
