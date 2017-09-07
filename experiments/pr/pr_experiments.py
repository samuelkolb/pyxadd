from __future__ import print_function

import json
import random

import os
import numpy
import scipy.sparse as sparse
import scipy.stats as stats
import time

import setup
import pyxadd.diagram as core

from experiments.citations.evaluation import KendallTau, TopInclusion, EvaluationMeasure, TopKT
from experiments.citations.parse import read_xadds
from experiments.link_prediction import learn_decision_tree, decision_tree_to_xadd, export_classifier
from experiments.pagerank import pagerank

from pyxadd.build import Builder
from pyxadd.evaluate import mass_evaluate
from pyxadd.matrix.matrix import Matrix
from pyxadd.matrix_vector import sum_out
from pyxadd.timer import Timer


def head(sequence):
    return sequence[0]


def get_bucket_indices(folds, size):
    import math
    indices = list(range(size))
    random.shuffle(indices)
    buckets = []
    step = int(math.ceil(size / float(folds)))
    for i in range(folds):
        start = i * step
        stop = min(len(indices) - 1, (i + 1) * step - 1)
        buckets.append(indices[start:stop])
    return buckets


def export_ground_pagerank(values_ground, path):
    with open(path, "w") as f:
        for author_value in values_ground:
            print(author_value, file=f)


def import_ground_values(path):
    with open(path, "r") as f:
        values = []
        for line in f:
            values.append(float(line.rstrip("\n")))
        return values


def sort_nodes(nodes, values):
    combined = zip(nodes, values)
    sorted_nodes = sorted(combined, key=lambda t: t[0])
    sorted_both = sorted(sorted_nodes, key=lambda t: t[1], reverse=True)
    return [name for name, _ in sorted_both]


def make_histogram(values):
    histogram = {}
    for val in values:
        if val not in histogram:
            histogram[val] = 0
        histogram[val] += 1
    return histogram


def count_links(edges):
    count = 0
    for i in range(len(edges)):
        count += len(edges[i])
    return count


def copy_edges(edges, copy_rate):
    new_list = []
    for a1 in range(len(edges)):
        new_edges = set(edges[a1])
        for n1 in edges[a1]:
            for n2 in edges[n1]:
                if random.random() < copy_rate:
                    new_edges.add(n2)
        new_edges = sorted(list(new_edges))
        new_list.append(new_edges)
    return new_list


class DataSet(object):
    """
    A data-set stores attributes and edges in relative format (indices are in range(0, len(attributes)).
    If necessary, absolute indices can be reconstructed using the node_keys which map relative to absolute indices.
    """

    def __init__(self, node_keys, attributes, edges):
        assert len(node_keys) == len(attributes) == len(edges)
        self.node_keys = node_keys
        self.attributes = attributes
        self.edges = edges

    @property
    def link_count(self):
        return count_links(self.edges)

    def get_random_subset(self, size):
        if size < len(self):
            return self.get_subset(sorted(random.sample(range(len(self)), size)))
        return self

    def get_subset(self, indices):
        """
        Computes a subset of the data set that only includes the authors corresponding to the given indices
        :param List[int] indices: The indices of the authors to include
        :return DataSet: The subset data set
        """
        index_set = set(indices)
        node_keys = [self.node_keys[i] for i in indices]
        attributes = [self.attributes[i] for i in indices]
        lookup = {indices[i]: i for i in range(len(node_keys))}
        edges = [[lookup[n] for n in self.edges[i] if n in index_set] for i in indices]
        return DataSet(node_keys, attributes, edges)

    def __len__(self):
        return len(self.attributes)

    def get_attributes(self, indices=None):
        return self.attributes if indices is None else dict(zip(indices, self.attributes[indices]))

    def copy_edges(self, copy_rate):
        """
        Increases the amount of edges by including 2nd degree neighbors as direct edges with a probability if copy_rate.
        :param float copy_rate: The probability of including a 2nd degree neighbor
        :return DataSet: A data set object with the new edges
        """
        edges = copy_edges(self.edges, copy_rate)
        return DataSet(self.node_keys, self.attributes, edges)

    def reload(self, timer, location, identifier):
        self.export_to_file(timer, location, identifier)
        return DataSet.import_from_disk(timer, location, identifier)

    # def as_list(self):
    #     return [self.authors, self.neighbors, self.sum_papers, self.median_years]

    def get_buckets(self, folds):
        """
        Partitions the indices into folds of training and test data.
        :param int folds: The number of folds to produce
        :return List[Tuple[List[int], List[int]]]: A list of folds, every fold is a tuple of training and test indices
        """
        bucket_indices = get_bucket_indices(folds, len(self))
        buckets = []
        for i in range(folds):
            training_indices = []
            testing_indices = None
            for j in range(folds):
                if i != j:
                    training_indices += bucket_indices[j]
                else:
                    testing_indices = bucket_indices[j]
            buckets.append((sorted(training_indices), sorted(testing_indices)))
        return buckets

    @staticmethod
    def get_file_name(location, name, identifier=None):
        suffix = "" if identifier is None else "_{}".format(identifier)
        return "{}/{}{}.txt".format(location, name, suffix)

    def export_to_file(self, timer, location, identifier):
        names = ["attributes", "edges"]
        attr_filename, edges_filename = (self.get_file_name(location, name, identifier) for name in names)

        timer.start("Exporting attributes to {}".format(attr_filename))
        with open(attr_filename, "w") as f:
            for i in range(len(self.node_keys)):
                print("{}, {}".format(self.node_keys[i], ", ".join(map(str, self.attributes[i]))), file=f)

        timer.start("Exporting edges to {}".format(edges_filename))
        with open(edges_filename, "w") as f:
            for i in range(len(self)):
                print(", ".join(map(str, self.edges[i])), file=f)

        timer.stop()

    @staticmethod
    def import_from_disk(timer, location, identifier=None):
        names = ["attributes", "edges"]
        attr_filename, edges_filename = (DataSet.get_file_name(location, name, identifier) for name in names)

        node_keys = []
        attributes = []
        edges = []

        timer.start("Importing attributes from {}".format(attr_filename))
        with open(attr_filename) as f:
            for line in f:
                line = line.rstrip("\n")
                if line != "":
                    entries = map(int, line.split(", "))
                    node_keys.append(entries[0])
                    attributes.append(list(entries[1:]))

        timer.start("Importing edges from {}".format(edges_filename))
        with open(edges_filename) as f:
            i = 0
            for line in f:
                if i < len(attributes):
                    i += 1
                    line = line.rstrip("\n")
                    entries = [] if line == "" else map(int, line.split(", "))
                    edges.append(list(entries))

        timer.stop()
        return DataSet(node_keys, attributes, edges)


class Experiment(object):
    def __init__(self, config, delta):
        """
        Initializes a single experiment
        :param setup.ExpConfig config: The PageRank configuration object
        :param float delta: The tolerance
        """
        self.config = config
        self.delta = delta

    def xadd_page_rank(self, timer, matrix, variables, iterations, diagram_file=None, diagram_export_file=None):
        """
        Computes the PageRank for an xadd-matrix
        :param Timer timer: A timer object
        :param Matrix matrix: The xadd-matrix to compute the PageRank for
        :param List[Tuple[str, int, int]] variables: Tuples of variable names, domain-lower- and -upper-bounds
        :param int iterations: Number of iterations
        :param str diagram_file: An optional file to print a graphical representation of the matrix to
        :param str diagram_export_file: An optional file to export the matrix to
        :return Tuple[Matrix, float]: A tuple of the result and the time taken
        """
        damping_factor = self.config.damping_factor
        delta = self.delta

        if diagram_file is not None:
            timer.start("Exporting diagram to {}".format(diagram_file))
            matrix.export(diagram_file)

        if diagram_export_file is not None:
            timer.start("Exporting diagram pool to {} (root: {}, row-vars: {}, col-vars: {})"
                        .format(diagram_export_file, matrix.diagram.root_id, matrix.row_variables,
                                matrix.column_variables))
            from pyxadd.diagram import Pool
            with open(diagram_export_file, "w") as stream:
                stream.write(Pool.to_json(matrix.diagram.pool))

        timer.start("Computing lifted pagerank")
        converged, iterations = pagerank(matrix, damping_factor, variables, delta=delta, norm=1, iterations=iterations)
        timer.log("Converged after {} iterations".format(iterations))
        return converged, timer.stop()

    def sparse_ground_page_rank(self, timer, data_set, iterations):
        """
        Computes the ground PageRank for the given data set
        :param Timer timer: A timer object
        :param DataSet data_set: The data set to compute the PageRank for
        :param int iterations: The maximal number of iteration
        :return List[float]: The PageRank value be node in the data set
        """
        damping_factor = self.config.damping_factor
        delta = self.delta

        timer.start("Counting links")
        count = count_links(data_set.edges)

        n = len(data_set)
        timer.log("{} links, density: {}".format(count, count / float(n * n)))

        timer.start("Creating sparse adjacency matrix")
        row = numpy.zeros(count)
        col = numpy.zeros(count)
        data = numpy.zeros(count)
        index = 0
        for i in range(n):
            for j in data_set.edges[i]:
                row[index] = i
                col[index] = j
                data[index] = 1
                # adjacency_matrix[i, j] = 1
                index += 1

        from scipy.sparse import coo_matrix
        # from experiments.citations.sparse import simple
        adjacency_matrix = coo_matrix((data, (row, col)), shape=(n, n), dtype=numpy.float)
        # timer.start("Computing sparse page-rank")
        # values_ground, iterations = simple(adjacency_matrix, damping_factor, delta, iterations, norm=1)

        from experiments.citations.sparse_pagerank_networkx import pagerank_scipy
        timer.start("Computing reference page-rank (networkx)")
        ground_values = pagerank_scipy(adjacency_matrix, alpha=damping_factor, max_iter=iterations, tol=delta)
        return ground_values


def get_checked_matrix(diagram, row_variables, col_variables):
    """
    Converts a diagram to a matrix that additionally checks the domain variables
    :param pyxadd.diagram.Diagram diagram: The diagram to convert
    :param List[Tuple[str, int, int]] row_variables: The domain row variables
    :param List[Tuple[str, int, int]] col_variables: The domain column variables
    :return Matrix: The matrix corresponding to the diagram with added domain checks
    """
    build = Builder(diagram.pool)
    for var, lb, ub in row_variables + col_variables:
        diagram *= build.limit(var, lb, ub)
    return Matrix(diagram, row_variables, col_variables)


def learn_dt(timer, training_set, options, tree_file, sample_count=None):
    """
    Computes a decision tree
    :param Timer timer: A timer object
    :param DataSet training_set: The training_set
    :param dict options: Options to be passed to the learner
    :param str tree_file: Optional filename where the tree should be stored
    :param int sample_count: Optional number of samples to use (default is all)
    :return:
    """
    timer.start("Computing learning examples and labels")
    examples = []

    for i in range(len(training_set)):
        for n in training_set.edges[i]:
            examples.append((i, n))

    if sample_count is not None and sample_count < len(examples):
        examples = random.sample(examples, sample_count)

    required = 2 * len(examples)
    labels = [1] * len(examples) + [0] * len(examples)

    forbidden = set(examples)
    while len(examples) < required:
        i = random.randint(0, len(training_set) - 1)
        j = random.randint(0, len(training_set) - 1)
        if i != j and (i, j) not in forbidden:
            examples.append((i, j))
            forbidden.add((i, j))

    examples = [training_set.attributes[i] + training_set.attributes[j] for i, j in examples]

    timer.start("Learning decision tree")
    clf = learn_decision_tree(examples, labels, options)

    if tree_file is not None:
        timer.start("Exporting decision tree to {}".format(tree_file))
        export_classifier(clf, tree_file)

    return clf


def ground_vector(vector, var_names, attributes):
    """
    Computes the PageRank values for a given set of attributes
    :param Matrix vector: The xadd-vector (PageRank result)
    :param List[str] var_names: The attribute variable names
    :param List[List[int]] attributes: The attribute vectors
    :return List[float]: A list of values, one for each attribute vector
    """
    assignments = [{var_names[j]: attributes[i][j] for j in range(len(var_names))} for i in range(len(attributes))]
    return mass_evaluate(vector.diagram, assignments)


def compare_decision_trees(true_dt, learned_dt):
    """
    Compares a true and a learned decision tree, returns TP, FP, TN, FN.
    :param Matrix true_dt: The ground-truth decision tree (GT)
    :param Matrix learned_dt: The leaned decision tree
    :return Tuple[int, int, int, int]: True positives, false positives, true negatives, false negatives
    """

    def count(diagram):
        """
        Sums out all variables from the given diagram
        :param pyxadd.diagram.Diagram diagram: The diagram
        :return: The count
        """
        return diagram.pool.diagram(sum_out(diagram.pool, diagram.root_id, map(head, variables))).evaluate({})

    variables = true_dt.row_variables + true_dt.column_variables
    build = Builder(true_dt.diagram.pool)
    bounds = build.exp(1)
    for var, lb, ub in variables:
        bounds &= build.limit(var, lb, ub)
    positives = count(learned_dt.diagram)
    negatives = count(~learned_dt.diagram & bounds)
    true_positive = count(true_dt.diagram & learned_dt.diagram)
    true_negative = count(~true_dt.diagram & ~learned_dt.diagram & bounds)
    return true_positive, positives - true_positive, true_negative, negatives - true_negative


class ExperimentRunner(object):
    def __init__(self, config_filename, tree_filename, output_dir="temp", input_files_directory=None):
        with open(config_filename) as stream:
            self.variables = json.load(stream)["variables"]

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.experiments = []
        self.timer = Timer()
        self._full_data_set = None
        self.input_files_directory = input_files_directory
        self.model_tree_file = tree_filename

    @property
    def full_data_set(self):
        """
        :rtype: DataSet
        """
        self.load_data_set()
        return self._full_data_set

    def load_data_set(self):
        if self._full_data_set is None:
            input_files_directory = self.input_files_directory if self.input_files_directory is not None else "."
            self._full_data_set = DataSet.import_from_disk(self.timer, input_files_directory)

    def run(self, size, delta, iterations, damping_factor, copy_rate, discrete, tree_depth, leaf_cutoff_rate, folds,
            top_count, learning_sample_count):

        self.load_data_set()
        data_size = len(self.full_data_set)
        if data_size < size:
            size = data_size

        attr_vars = self.variables
        row_vars = [("r_{}".format(name), lb, ub) for name, lb, ub in attr_vars]
        col_vars = [("c_{}".format(name), lb, ub) for name, lb, ub in attr_vars]

        # pool_file = "temp/pool_{}.txt".format(size)

        model_tree_file = self.model_tree_file
        model_tree_export_file = "{}/model_tree.dot".format(self.output_dir)

        cache_ground_values = False

        timer = Timer()

        setting = setup.ExpConfig(size, copy_rate, damping_factor, tree_depth, iterations, leaf_cutoff_rate, 100, folds,
                                  top_count, learning_sample_count)

        options = {"max_depth": tree_depth, "min_samples_leaf": int(size * leaf_cutoff_rate)}
        subset = self.full_data_set.get_random_subset(size).reload(timer, self.output_dir, size)

        if copy_rate > 0:
            timer.start("Copy edges (copy rate={})".format(copy_rate))
            subset = subset.copy_edges(copy_rate)

        timer.start("Computing buckets ({} folds)".format(setting.folds))
        buckets = subset.get_buckets(setting.folds)

        # Below is the iterative part (per bucket)

        experiments = []
        measures = [(KendallTau(), "kt"), (TopInclusion(top_count), "top_inclusion"), (TopKT(top_count), "kt_top")]
        verification = setting.verification_iterations

        for i in range(len(buckets)):
            tree_file = "{}/tree_{}_{}.dot".format(self.output_dir, size, i)
            diagram_file = "{}/diagram_{}_{}".format(self.output_dir, size, i)
            converged_file = "{}/converged_{}_{}".format(self.output_dir, size, i)
            ground_training_values_file = "{}/ground_value_same_{}_{}.txt".format(self.output_dir, size, i)
            values_ground_file = "{}/ground_value_{}_{}.txt".format(self.output_dir, size, i)
            ground_all_values_file = "{}/full_ground_value_same_{}_{}.txt".format(self.output_dir, size, i)
            # difference_tree_file = "{}/difference_{}_{}.txt".format(self.directory, size, i)

            training_indices, testing_indices = buckets[i]
            training_set = subset.get_subset(training_indices)
            test_set = subset.get_subset(testing_indices)

            exp_results = setting.get_experiment()

            timer.start("Counting links")
            exp_results.links = subset.link_count
            timer.stop()

            experiment = Experiment(setting, delta)
            pool = core.Pool()
            for var in map(head, attr_vars + row_vars + col_vars):
                pool.add_var(var, "int")

            # Computations for true model ###
            timer.start("Computing true values")
            true_timer = timer.sub_time()

            true_timer.start("Importing true model")
            true_matrix = get_checked_matrix(read_xadds(model_tree_file, pool)[0], row_vars, col_vars).reduce()

            true_timer.start("Calculating lifted pagerank (true with {} iterations)".format(verification))
            true_vector, true_pr_time = experiment.xadd_page_rank(true_timer, true_matrix, attr_vars, verification)

            true_timer.start("Computing true values for training authors")
            true_training_values = ground_vector(true_vector, map(head, attr_vars), training_set.attributes)

            true_timer.start("Computing true values for test authors")
            true_test_values = ground_vector(true_vector, map(head, attr_vars), test_set.attributes)

            true_timer.start("Exporting model tree")
            true_matrix.export(model_tree_export_file)

            # Computations for lifted model ###
            timer.start("Computing lifted values")
            lifted_timer = timer.sub_time()

            lifted_timer.start("Learning decision tree")
            learned_dt = learn_dt(lifted_timer.sub_time(), training_set, options, tree_file, learning_sample_count)
            lifted_timer.sub_time().start("Converting decision tree to XADD")
            xadd = decision_tree_to_xadd(pool, learned_dt, row_vars + col_vars, discrete=discrete)
            lifted_matrix = Matrix(xadd, row_vars, col_vars).reduce()
            exp_results.speed_learning_lifted = lifted_timer.stop()

            lifted_timer.start("Calculating lifted pagerank (lifted with {} iterations)".format(iterations))
            lifted_vector, lifted_pr_time = experiment.xadd_page_rank(lifted_timer, lifted_matrix, attr_vars, iterations, diagram_file)
            exp_results.speed_pagerank_lifted = lifted_pr_time

            lifted_timer.start("Computing lifted values for training authors")
            lifted_training_values = ground_vector(lifted_vector, map(head, attr_vars), training_set.attributes)
            exp_results.speed_grounding_lifted = lifted_timer.stop()

            exp_results.speed_lifted = timer.read()

            lifted_timer.start("Computing lifted values for test authors")
            lifted_test_values = ground_vector(lifted_vector, map(head, attr_vars), test_set.attributes)

            lifted_timer.start("Exporting converged diagram to {}".format(converged_file))
            lifted_vector.export(converged_file)

            # Computations on decision tree accuracy ###
            timer.start("Computing decision tree accuracy")
            tp, fp, tn, fn = compare_decision_trees(true_matrix, lifted_matrix)
            timer.log("TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}".format(tp=tp, fp=fp, tn=tn, fn=fn))
            exp_results.true_positive = tp
            exp_results.false_positive = fp
            exp_results.true_negative = tn
            exp_results.false_negative = fn

            # Computations for ground pagerank ###
            timer.start("Computing ground values")
            ground_timer = timer.sub_time()
            if not cache_ground_values or not os.path.isfile(ground_training_values_file) \
                    or not os.path.isfile(values_ground_file):

                ground_timer.start("Calculating training pagerank (ground with {} iterations)".format(iterations))
                ground_training_values = experiment.sparse_ground_page_rank(ground_timer, training_set, iterations)
                exp_results.speed_ground = ground_timer.stop()

                ground_timer.start("Exporting training pagerank to {}".format(ground_training_values_file))
                export_ground_pagerank(ground_training_values, ground_training_values_file)

                ground_timer.start("Calculating full pagerank (ground with {} iterations)".format(iterations))
                ground_all_values = experiment.sparse_ground_page_rank(ground_timer, subset, iterations)

                ground_timer.start("Exporting full pagerank to {}".format(ground_all_values_file))
                export_ground_pagerank(ground_all_values, ground_all_values_file)

            ground_timer.start("Importing training pagerank from {}".format(ground_training_values_file))
            ground_training_values = import_ground_values(ground_training_values_file)

            ground_timer.start("Importing full pagerank from {}".format(ground_all_values_file))
            ground_all_values = import_ground_values(ground_all_values_file)
            ground_test_values = [ground_all_values[i] for i in testing_indices]

            # Computing evaluations ###
            timer.start("Evaluating methods")
            eval_timer = timer.sub_time()

            def get_attr_name(_eval_name, _method1, _method2, _data_set):
                return "{}_{}_{}_{}".format(_eval_name, _method1, _method2, _data_set)

            def eval_and_set(_evaluation, _eval_name, _timer, _method_name, _values, _true_values, _data_set):
                attr_name = get_attr_name(_eval_name, _method_name, "true", _data_set)
                setattr(exp_results, attr_name, _evaluation.evaluate(_timer, _values, _true_values))

            eval_timer.start("Computing constant baseline")
            constant_test_values = numpy.zeros(len(ground_test_values))
            constant_test_values[::2] = 1

            eval_timer.start("Computing random baseline")
            random_test_values = list(random.random() for _ in range(len(ground_test_values)))

            method_values = [
                ("ground", ground_training_values, true_training_values, "training"),
                ("lifted", lifted_training_values, true_training_values, "training"),
                ("ground", ground_test_values, true_test_values, "test"),
                ("lifted", lifted_test_values, true_test_values, "test"),
                ("constant", constant_test_values, true_test_values, "test"),
                ("random", random_test_values, true_test_values, "test"),
            ]

            for evaluation, eval_name in measures:
                assert isinstance(evaluation, EvaluationMeasure)
                eval_timer.start("Evaluating {} measure".format(eval_name))
                m_timer = eval_timer.sub_time()
                for m_name, m_values, t_values, ds_name in method_values:
                    m_timer.start("Evaluating {} values on the {} set".format(m_name, ds_name))
                    eval_and_set(evaluation, eval_name, eval_timer, m_name, m_values, t_values, ds_name)

            timer.stop()

            experiments.append(exp_results)

        self.experiments.append(experiments)
        return experiments

    def reset(self):
        self.experiments = []

    def export_experiments(self):
        output_file = "{}/output_{}.txt".format(self.output_dir, time.strftime("%Y%m%d_%H%M%S"))
        with open(output_file, "w") as stream:
            stream.write(json.dumps([map(lambda e: e.export_to_dict(), experiment) for experiment in self.experiments]))

    @staticmethod
    def import_experiments(output_dir, output_file):
        path = "{}/{}".format(output_dir, output_file)
        with open(path) as stream:
            experiment_list = json.load(stream)
            experiments = list()
            for experiment in experiment_list:
                if isinstance(experiment, list):
                    experiments.append(map(setup.ExpResults.import_from_dict, experiment))
                else:
                    experiments.append([setup.ExpResults.import_from_dict(experiment)])
        return experiments
