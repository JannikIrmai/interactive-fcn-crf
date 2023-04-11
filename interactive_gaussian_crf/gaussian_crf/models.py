import numpy as np
import networkx as nx
from abc import ABC, abstractmethod


class CRFBase(ABC):

    @abstractmethod
    def get_dim(self):
        pass

    @abstractmethod
    def get_n(self):
        pass

    @abstractmethod
    def get_mean(self, i: int):
        pass

    @abstractmethod
    def set_mean(self, i: int, mean: np.ndarray):
        pass

    @abstractmethod
    def get_precision(self, i: int):
        pass

    @abstractmethod
    def set_precision(self, i: int, precision: np.ndarray):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def infer_with_evidence(self, evidence: dict):
        pass

    @abstractmethod
    def fit_unary_from_heatmap(self, heatmap: np.ndarray):
        pass

    @abstractmethod
    def fit(self, locations: np.ndarray):
        pass

    @abstractmethod
    def get_variables(self):
        pass

    @abstractmethod
    def get_edges(self):
        pass


class HigherOrderGaussianCRF(CRFBase):

    def __init__(self, variables: (list, int), edges: list, dim: int):

        self._dim = dim

        if isinstance(variables, int):
            variables = list(range(variables))
        self._var2idx = {n: i for i, n in enumerate(variables)}

        self._n = len(variables)

        self._edges = list()
        for e, a in edges:
            for v in e:
                if v not in self._var2idx:
                    raise KeyError(f"Variable {v} of edge {e} not in variables.")
            self._edges.append((e, a, np.zeros(self._dim), np.eye(self._dim)))
            
        # hidden values
        self._combined_precision = None
        self._combined_mean = None

    def get_dim(self):
        return self._dim

    def get_variables(self):
        return [v for v in self._var2idx]

    def get_edges(self):
        return [e for e in self._edges]

    def get_n(self):
        return self._n

    def set_mean(self, e: int, mean: np.ndarray):
        self._edges[e][2][:] = mean
        self._combined_mean = None
        self._combined_precision = None

    def set_precision(self, e: int, precision: np.ndarray):
        self._edges[e][3][:] = precision
        self._combined_mean = None
        self._combined_precision = None

    def get_mean(self, e: int):
        return self._edges[e][2].copy()

    def get_precision(self, e: int):
        return self._edges[e][3].copy()

    def fit_unary_from_heatmap(self, heatmap: np.ndarray):
        assert heatmap.ndim == self._dim + 1
        assert heatmap.shape[0] == self._n

        coordinates = np.array(np.unravel_index(np.arange(np.product(heatmap.shape[1:])), heatmap.shape[1:]))
        for e, a, mean, precision in self._edges:
            if len(e) > 1:
                continue
            try:
                i = self._var2idx[e[0]]
                flat = heatmap[i].flatten()
                idx = flat > 0
                mean[:] = np.average(coordinates[:, idx], axis=1, weights=flat[idx])
                cov = np.cov(coordinates[:, idx], ddof=0, aweights=flat[idx])
                precision[:] = np.linalg.inv(cov)
            except (np.linalg.LinAlgError, ZeroDivisionError):
                mean[:] = 0
                precision[:] = 0
        self._combined_precision = None
        self._combined_mean = None

    def fit(self, locations: np.ndarray):
        assert locations.shape[1:] == (self._n, self._dim)

        for e, a, mean, precision in self._edges:
            lin_comb = np.zeros((locations.shape[0], self._dim))
            for e_i, a_i in zip(e, a):
                i = self._var2idx[e_i]
                lin_comb += a_i * locations[:, i]
            # remove the nan rows
            lin_comb = lin_comb[~np.any(np.isnan(lin_comb), axis=1)]
            mean[:] = np.mean(lin_comb, axis=0)
            try:
                precision[:] = np.linalg.inv(np.cov(lin_comb.T))
            except np.linalg.LinAlgError:
                precision[:] = 0
        self._combined_precision = None
        self._combined_mean = None
                
    def get_combined_precision(self):
        if self._combined_precision is not None:
            return self._combined_precision
        # create the combined precision matrix
        self._combined_precision = np.zeros((self._n * self._dim, self._n * self._dim))
        for v, i in self._var2idx.items():
            for w, j in self._var2idx.items():
                sum_of_precisions = np.zeros((self._dim, self._dim))
                for e, a, mean, precision in self._edges:
                    if v in e and w in e:
                        sum_of_precisions += a[e.index(v)] * a[e.index(w)] * precision
                self._combined_precision[self._dim*i: self._dim*(i+1), self._dim*j: self._dim*(j+1)] = sum_of_precisions
        return self._combined_precision

    def get_combined_mean(self):
        if self._combined_mean is not None:
            return self._combined_mean
        # create the vector b
        b = np.zeros(self._dim * self._n)
        for v, i in self._var2idx.items():
            b_v = np.zeros(self._dim)
            for e, a, mean, precision in self._edges:
                if v in e:
                    b_v += a[e.index(v)] * precision.dot(mean)
            b[self._dim * i: self._dim * (i + 1)] = b_v
        # compute combined mean
        cov = np.linalg.inv(self.get_combined_precision())
        self._combined_mean = cov.dot(b)
        return self._combined_mean

    def get_log_det(self):
        combined_precision = self.get_combined_precision()
        cholesky = np.linalg.cholesky(combined_precision)
        return 2 * np.sum(np.log(np.diag(cholesky)))

    def infer(self):
        cov = np.linalg.inv(self.get_combined_precision())
        combined_mean = self.get_combined_mean()
        # extract the means and covariances of the individual variables
        cov_dict = dict()
        mean_dict = dict()
        for v, i in self._var2idx.items():
            cov_dict[v] = cov[self._dim * i:self._dim * (i + 1), self._dim * i:self._dim * (i + 1)]
            mean_dict[v] = combined_mean[self._dim * i: self._dim*(i+1)]

        return mean_dict, cov_dict

    def infer_with_evidence(self, evidence: dict):
        # TODO this method could probably be implemented without creating a new instance
        if len(evidence) == 0:
            return self.infer()
        if len(evidence) == self._n:
            return evidence, dict()
        # construct a new HigherOrderGaussianCRF instance that includes only the variables that do not have evidence
        remaining_variables = [v for v in self._var2idx if v not in evidence]
        remaining_edges = []
        for e, a, mean, precision in self._edges:
            remaining_e = tuple(v for v in e if v in remaining_variables)
            if len(remaining_e) == 0:
                continue
            remaining_a = tuple(a_v for v, a_v in zip(e, a) if v in remaining_e)
            remaining_mean = mean.copy()
            for v, a_v in zip(e, a):
                if v not in remaining_e:
                    remaining_mean -= a_v * evidence[v]
            remaining_edges.append((remaining_e, remaining_a, remaining_mean, precision))
        crf = HigherOrderGaussianCRF(variables=remaining_variables, edges=[], dim=self._dim)
        crf._edges = remaining_edges
        # infer the remaining variables
        means, cov = crf.infer()
        for v, ev in evidence.items():
            means[v] = ev
        return means, cov

    def log_likelihood(self, evidence: dict):
        ll = 1/2 * self.get_log_det()
        for e, a, mean, precision in self._edges:
            lin_comb = - mean.copy()
            for v, a_v in zip(e, a):
                lin_comb += a_v * evidence[v]
            ll += -1/2 * lin_comb.dot(precision.dot(lin_comb))
        return ll - 1/2 * self._n * self._dim * np.log(2*np.pi)


class GaussianCRF(CRFBase):

    def __init__(self, graph: nx.Graph = None, n: int = None, dim: int = 1):

        if graph is None and n is None:
            raise ValueError("Specify at least graph or num_variables")
        if graph is not None:
            if n is not None and graph.number_of_nodes() != n:
                raise ValueError("Number of nodes in graph does not agree with num_variables")
            self._graph = graph.copy()
        else:
            self._graph = nx.complete_graph(n)

        self._node2idx = {v: i for i, v in enumerate(self._graph.nodes)}

        self._n = self._graph.number_of_nodes()

        self._dim = dim
        self._unary_means = {n: np.zeros(dim) for n in self._graph.nodes()}
        self._unary_precisions = {e: np.eye(dim) for e in self._graph.nodes()}
        self._pairwise_means = {n: np.zeros(dim) for n in self._graph.edges()}
        self._pairwise_precisions = {e: np.eye(dim) for e in self._graph.edges()}

    def get_n(self):
        return self._n

    def get_variables(self):
        return [v for v in self._graph.nodes]

    def get_edges(self):
        return [e for e in self._graph.edges]

    def get_dim(self):
        return self._dim

    def get_precision(self, idx):
        if idx in self._unary_precisions:
            return self._unary_precisions[idx].copy()
        else:
            return self._pairwise_precisions[idx].copy()

    def set_mean(self, i, mean: np.ndarray):
        if i in self._unary_means:
            self._unary_means[i][:] = mean
        else:
            self._pairwise_means[i][:] = mean

    def set_precision(self, i, precision: np.ndarray):
        if i in self._unary_precisions:
            self._unary_precisions[i][:] = precision
        else:
            self._pairwise_precisions[i][:] = precision

    def get_mean(self, idx):
        if idx in self._unary_means:
            return self._unary_means[idx].copy()
        else:
            return self._pairwise_means[idx].copy()

    def fit(self, locations: np.ndarray):
        assert locations.shape[1:] == (self._n, self._dim)

        for i, j in self._graph.edges:
            diff = locations[:, i] - locations[:, j]
            diff = diff[~np.any(np.isnan(diff), axis=1)]
            self._pairwise_means[(i, j)] = np.mean(diff, axis=0)
            try:
                precision = np.linalg.inv(np.cov(diff.T))
            except np.linalg.LinAlgError:
                precision = np.zeros((self._dim, self._dim))
            self._pairwise_precisions[(i, j)] = precision

    def fit_unary_from_heatmap(self, heatmap: np.ndarray):
        assert heatmap.ndim == self._dim + 1
        assert heatmap.shape[0] == self._n

        coordinates = np.array(np.unravel_index(np.arange(np.product(heatmap.shape[1:])), heatmap.shape[1:]))
        for i in range(self._n):
            try:
                flat = heatmap[i].flatten()
                idx = flat > 0
                self._unary_means[i] = np.average(coordinates[:, idx], axis=1, weights=flat[idx])
                cov = np.cov(coordinates[:, idx], ddof=0, aweights=flat[idx])
                self._unary_precisions[i] = np.linalg.inv(cov)
            except (np.linalg.LinAlgError, ZeroDivisionError):
                self._unary_means[i] = np.zeros(self._dim)
                self._unary_precisions[i] = np.zeros((self._dim, self._dim))

    def infer(self):
        # create the combined precision matrix
        combined_precision = np.zeros((self._n * self._dim, self._n * self._dim))
        for v in self._graph.nodes:
            sum_of_precisions = self._unary_precisions[v].copy()
            for n in self._graph.neighbors(v):
                if (v, n) in self._pairwise_precisions:
                    sum_of_precisions += self._pairwise_precisions[v, n]
                else:
                    sum_of_precisions += self._pairwise_precisions[n, v]
            i = self._node2idx[v]
            combined_precision[self._dim * i: self._dim * (i + 1), self._dim * i: self._dim * (i + 1)] = \
                sum_of_precisions
        for v, w in self._graph.edges:
            i = self._node2idx[v]
            j = self._node2idx[w]
            combined_precision[self._dim * i: self._dim * (i + 1), self._dim * j: self._dim * (j + 1)] = \
                - self._pairwise_precisions[v, w]
            combined_precision[self._dim * j: self._dim * (j + 1), self._dim * i: self._dim * (i + 1)] = \
                - self._pairwise_precisions[v, w]
        # create the vector b
        b = np.zeros(self._dim * self._n)
        for v in self._graph.nodes:
            b_v = self._unary_precisions[v].dot(self._unary_means[v])
            for n in self._graph.neighbors(v):
                if (v, n) in self._pairwise_means:
                    b_v += self._pairwise_precisions[v, n].dot(self._pairwise_means[v, n])
                else:
                    b_v -= self._pairwise_precisions[n, v].dot(self._pairwise_means[n, v])
            i = self._node2idx[v]
            b[self._dim * i: self._dim * (i + 1)] = b_v
        # compute combined mean
        cov = np.linalg.inv(combined_precision)
        combined_mean = cov.dot(b)
        cov_dict = dict()
        mean_dict = dict()
        for v in self._graph.nodes:
            i = self._node2idx[v]
            cov_dict[v] = cov[self._dim*i:self._dim*(i+1), self._dim*i:self._dim*(i+1)]
            mean_dict[v] = combined_mean[self._dim * i: self._dim * (i + 1)]

        # return variable wise mean and covariance
        return mean_dict, cov_dict

    def infer_with_evidence(self, evidence: dict):
        if len(evidence) == 0:
            return self.infer()
        if len(evidence) == self._n:
            return evidence, dict()

        # construct a new gaussian CRF that represent the variables that need to be inferred
        graph_without_evidence = self._graph.copy()
        for v in evidence:
            graph_without_evidence.remove_node(v)

        crf_with_evidence = GaussianCRF(graph_without_evidence, dim=self._dim)
        for v in graph_without_evidence.nodes:
            precision = self._unary_precisions[v].copy()
            mean = self._unary_precisions[v].dot(self._unary_means[v])
            for n in self._graph.neighbors(v):
                if n in evidence:
                    if (v, n) in self._pairwise_precisions:
                        precision += self._pairwise_precisions[v, n]
                        mean += self._pairwise_precisions[v, n].dot(self._pairwise_means[v, n] + evidence[n])
                    else:
                        precision += self._pairwise_precisions[n, v]
                        mean += self._pairwise_precisions[n, v].dot(evidence[n] - self._pairwise_means[n, v])
            mean = np.linalg.inv(precision).dot(mean)
            crf_with_evidence._unary_means[v] = mean
            crf_with_evidence._unary_precisions[v] = precision
        for v, w in graph_without_evidence.edges:
            crf_with_evidence._pairwise_means[v, w] = self._pairwise_means[v, w]
            crf_with_evidence._pairwise_precisions[v, w] = self._pairwise_precisions[v, w]

        mean_dict, cov_dict = crf_with_evidence.infer()
        for v, ev in evidence.items():
            mean_dict[v] = ev
        return mean_dict, cov_dict
