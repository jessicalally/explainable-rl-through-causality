import castle.algorithms
from castle.common.priori_knowledge import PrioriKnowledge
from causalnex.structure.notears import from_numpy
import lingam
from lingam import VARLiNGAM
from lingam.utils import make_dot, print_dagc
import networkx as nx
import numpy as np


class Method(object):
    def generate_causal_matrix(
            self,
            data,
            forbidden_edges,
            required_edges,
            with_assumptions=True):
        raise NotImplementedError

    # TODO: add vcn as an option here
    @staticmethod
    def get_method(method):
        if method == 'pc':
            return PC()
        elif method == 'rl':
            return RL()
        elif method == 'notears':
            return NOTEARS()
        elif method == 'varlingam':
            return VarLiNGAM()
        elif method == 'directlingam':
            return DirectLiNGAM()
        else:
            raise ValueError(
                f'{method} is an unsupported causal discovery method')

    def threshold(self, causal_matrix, threshold):
        thresholded_causal_matrix = causal_matrix

        for i in range(len(causal_matrix)):
            for j in range(len(causal_matrix)):
                thresholded_causal_matrix[i][j] = 1 if abs(
                    causal_matrix[i][j]) > threshold else 0

        return thresholded_causal_matrix

    def post_process(self, causal_matrix, forbidden_edges, required_edges):
        processed_causal_matrix = causal_matrix

        for (i, j) in forbidden_edges:
            # Delete
            processed_causal_matrix[i][j] = 0

            # Flip
            # if processed_causal_matrix[i][j] == 1:
            #     # Flip forbidden edges, since these are backwards in time
            #     processed_causal_matrix[i][j] = 0
            #     processed_causal_matrix[j][i] = 1

        for (i, j) in required_edges:
            processed_causal_matrix[i][j] = 1

        return processed_causal_matrix


class PC(Method):
    def generate_causal_matrix(self, data, env, forbidden_edges, required_edges, with_assumptions=True):
        priori = PrioriKnowledge(data.shape[1])

        # PC does not require post-processing, as the prior information can be
        # given directly to the algorithm
        if with_assumptions:
            priori.add_forbidden_edges(forbidden_edges)
            priori.add_required_edges(required_edges)

        pc = castle.algorithms.PC(
            variant='stable',
            priori_knowledge=priori,
            ci_test='fisherz',
            alpha=0.005
        )

        pc.learn(data)

        return pc.causal_matrix


class RL(Method):
    def generate_causal_matrix(self, data, env, forbidden_edges, required_edges, with_assumptions=True):
        rl = castle.algorithms.RL(
            nb_epoch=1500,
            input_dimension=32,
            hidden_dim=32,
            num_heads=8,
            num_stacks=4
        )

        rl.learn(data)
        causal_matrix = rl.causal_matrix

        if with_assumptions:
            causal_matrix = self.post_process(
                causal_matrix, forbidden_edges, required_edges)

        return causal_matrix


class NOTEARS(Method):
    def generate_causal_matrix(self, data, env, forbidden_edges, required_edges, with_assumptions=True):
        threshold = 0.5

        nt = from_numpy(
            data,
            max_iter=100,
            h_tol=1e-08,
            w_threshold=0.5,
            tabu_edges=None,
            tabu_parent_nodes=None,
            tabu_child_nodes=None
        )

        causal_matrix = np.array(nx.adjacency_matrix(nt).todense())
        causal_matrix = self.threshold(causal_matrix, threshold)

        if with_assumptions:
            causal_matrix = self.post_process(
                causal_matrix, forbidden_edges, required_edges)

        return causal_matrix


class VarLiNGAM(Method):
    def plot_dag(self, adjacency_matrix, threshold, labels):
        dag = make_dot(
            adjacency_matrix,
            ignore_shape=True,
            lower_limit=threshold,
            labels=labels)
        dag.view()

    # Prints the probabilities of the top 3 estimated DAGs
    def print_dag_probabilities(self, model, data, threshold, labels):
        result = model.bootstrap(data, n_sampling=100)
        dagc = result.get_directed_acyclic_graph_counts(
            n_dags=3, min_causal_effect=threshold, split_by_causal_effect_sign=True)
        print_dagc(dagc, 100, labels=labels)

    def generate_causal_matrix(
            self,
            data,
            env,
            forbidden_edges,
            required_edges,
            threshold=0.3,
            with_assumptions=True,
            plot_dag=False,
            print_dag_probabilities=False,
            restructure=False):

        # Reshape data as VarLiNGAM currently requires a different CSV structure
        # than the other algorithms (rows of state + action)
        model = VARLiNGAM()
        model.fit(data)

        if restructure:
            causal_matrix = np.vstack(
                [
                    np.hstack(
                        [model.adjacency_matrices_[0].T,
                        model.adjacency_matrices_[1].T]
                    ),

                    np.hstack(
                        [np.zeros((data.shape[1], data.shape[1])),
                        model.adjacency_matrices_[0].T]
                    )
                ]
            )

            # TODO: explain this
            causal_matrix = np.delete(
                causal_matrix, len(causal_matrix) - 1, axis=0)
            causal_matrix = np.delete(
                causal_matrix, len(causal_matrix) - 1, axis=1)
            # causal_matrix = np.delete(causal_matrix, env.state_space + 1, axis=0)
            # causal_matrix = np.delete(causal_matrix, env.state_space + 1, axis=1)

        else:
            # TODO: not sure what this should be
            print(model.adjacency_matrices_[0].T)
            causal_matrix = model.adjacency_matrices_[0].T

        causal_matrix = self.threshold(causal_matrix, threshold)

        if with_assumptions:
            # VarLiNGAM already includes the assumption that causal relationships
            # that go backwards in time are invalid, because the algorithm is
            # designed for handling time series data
            causal_matrix = self.post_process(
                causal_matrix, forbidden_edges, required_edges)

        if plot_dag:
            self.plot_dag(
                np.hstack(
                    model.adjacency_matrices_),
                threshold,
                env.labels)

        if print_dag_probabilities:
            self.print_dag_probabilities(model, data, threshold, env.labels)

        return causal_matrix

class DirectLiNGAM(Method):
    def generate_causal_matrix(self, data, env, forbidden_edges, required_edges, with_assumptions=True):
        model = lingam.DirectLiNGAM()
        model.fit(data)

        causal_matrix = model.adjacency_matrix_
        causal_matrix = self.post_process(
                causal_matrix, forbidden_edges, required_edges)

        return causal_matrix