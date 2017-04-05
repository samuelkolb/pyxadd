from __future__ import print_function

import json
import random

import os
import numpy
import scipy.sparse as sparse
import scipy.stats as stats
import time

from experiments.citations.evaluation import KendallTau, TopInclusion, EvaluationMeasure, TopKT
from experiments.citations.parse import read_xadds
from experiments.link_prediction import learn_decision_tree, decision_tree_to_xadd, export_classifier
from experiments.pagerank import pagerank
from experiments.test import test_pagerank
from pyxadd.diagram import Pool
from pyxadd.evaluate import mass_evaluate
from pyxadd.matrix.matrix import Matrix

import sparse_pagerank
from pyxadd.optimization import find_optimal_conditions
from pyxadd.timer import Timer


def import_authors(path):
    authors = []
    lookup = dict()
    sum_papers = []
    with open(path) as f:
        for line in f:
            author, publication_count = line.rstrip("\n").split(", ")
            authors.append(author)
            lookup[author] = len(authors) - 1
            sum_papers.append(int(publication_count))
    return authors, lookup, sum_papers


def import_median_years(authors, path):
    lookup_median_years = dict()
    with open(path) as f:
        for line in f:
            author, median_year = line.rstrip("\n").split(", ")
            lookup_median_years[author] = int(median_year)
    return list(lookup_median_years[author] for author in authors)


def import_neighbors(authors, lookup, path):
    neighbors = [[] for _ in authors]
    with open(path) as f:
        for line in f:
            a1, a2 = line.rstrip("\n").split(", ")
            i1, i2 = lookup[a1], lookup[a2]
            neighbors[i1].append(i2)
            neighbors[i2].append(i1)

    neighbors = [list(set(n)) for n in neighbors]
    return neighbors


def export_subset(size, complete_authors, complete_neighbors, complete_sum_papers, complete_median_years, authors_file,
                  coauthors_file, median_years_file):
    if size < len(complete_authors):
        indices = random.sample(range(len(complete_authors)), size)
    else:
        indices = list(i for i in range(len(complete_authors)))

    subset = compute_subset(indices, complete_authors, complete_neighbors, complete_sum_papers, complete_median_years)
    subset_authors, subset_neighbors, subset_sum_papers, subset_median_years = subset

    with open(authors_file, "w") as f:
        for i in range(len(subset_authors)):
            print("{}, {}".format(subset_authors[i], subset_sum_papers[i]), file=f)

    with open(median_years_file, "w") as f:
        for i in range(len(subset_authors)):
            print("{}, {}".format(subset_authors[i], subset_median_years[i]), file=f)

    with open(coauthors_file, "w") as f:
        for i in range(len(subset_authors)):
            author_neighbors = subset_neighbors[i]
            for neighbor in author_neighbors:
                a1, a2 = sorted((subset_authors[i], subset_authors[neighbor]))
                print("{}, {}".format(a1, a2), file=f)


def compute_subset(indices, complete_authors, complete_neighbors, complete_sum_papers, complete_median_years):
    index_set = set(indices)
    subset_authors = [complete_authors[i] for i in indices]
    subset_lookup = {subset_authors[i]: i for i in range(len(subset_authors))}
    subset_sum_papers = [complete_sum_papers[i] for i in indices]
    subset_median_years = [complete_median_years[i] for i in indices]
    subset_neighbors = [
        [subset_lookup[complete_authors[n]] for n in author_neighbors if n in index_set]
        for author_neighbors in [complete_neighbors[i] for i in indices]
        ]
    return subset_authors, subset_neighbors, subset_sum_papers, subset_median_years


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


def copy_neighbors(neighbors, copy_rate):
    new_list = []
    for a1 in range(len(neighbors)):
        new_neighbors = set(neighbors[a1])
        for n1 in neighbors[a1]:
            for n2 in neighbors[n1]:
                if random.random() < copy_rate:
                    new_neighbors.add(n2)
        new_neighbors = sorted(list(new_neighbors))
        new_list.append(new_neighbors)
    return new_list


class AuthorPagerank(object):
    def __init__(self, authors, neighbors, sum_papers, median_years):
        # self.folds = folds

        self.authors = authors
        self.neighbors = neighbors
        self.sum_papers = sum_papers
        self.median_years = median_years

        self.attributes = None
        self.converged = None
        self.values = None
        self.variables = None
        self.clf = None
        self.learning_time = None
        self.pagerank_time = None
        self.grounding_time = None

    @property
    def attribute_count(self):
        return 3

    @property
    def variable_names(self):
        return ["f{}".format(i) for i in range(self.attribute_count)]

    def compute_decision_tree(self, timer, authors, attributes, neighbors, options, tree_file):
        timer.start("Computing learning examples and labels")
        examples = []
        labels = []

        for i in range(len(authors)):
            for n in neighbors[i]:
                if i < n:
                    examples.append((i, n))
                    labels.append(1)

        required = 2 * len(examples)
        forbidden = set(examples)
        while len(examples) < required:
            i = random.randint(0, len(authors) - 1)
            j = random.randint(0, len(authors) - 1)
            i, j = sorted((i, j))
            if i != j and (i, j) not in forbidden:
                examples.append((i, j))
                labels.append(0)
                forbidden.add((i, j))

        attribute_count = self.attribute_count
        max_attributes = list(0 for _ in range(attribute_count))
        min_attributes = list(0 for _ in range(attribute_count))
        for a in attributes:
            for i in range(attribute_count):
                if a[i] > max_attributes[i]:
                    max_attributes[i] = a[i]
                if a[i] < min_attributes[i]:
                    min_attributes[i] = a[i]

        examples = [attributes[i] + attributes[j] for i, j in examples]
        # print("\n".join(list(", ".join((str(e) for e in examples[i])) + ": " + str(labels[i]) for i in range(len(examples)))))

        timer.start("Learning decision tree")
        clf = learn_decision_tree(examples, labels, options)
        self.clf = clf

        if tree_file is not None:
            timer.start("Exporting decision tree to {}".format(tree_file))
            export_classifier(clf, tree_file)

        return clf, min_attributes, max_attributes

    def get_row_column_variables(self):
        diagram_variables = []
        for prefix in ("r", "c"):
            diagram_variables.append([("{}_{}".format(prefix, name), lb, ub) for name, lb, ub in self.variables])
        return tuple(diagram_variables)

    def compute_pagerank(self, timer, damping_factor, delta, iterations, tree_file=None, diagram_file=None,
                         diagram_export_file=None, discrete=True, options=None):
        if self.converged is not None:
            return

        authors, neighbors, sum_papers, median_years = self.authors, self.neighbors, self.sum_papers, self.median_years

        learning_time = 0

        timer.start("Computing attributes")
        sum_neighbors = []
        for i in range(len(authors)):
            sum_neighbors.append(len(neighbors[i]))

        attributes = zip(sum_papers, sum_neighbors, self.median_years)
        self.attributes = attributes
        # print(attributes)
        learning_time += timer.stop()

        timer.start("Learning decision tree")
        clf, min_attributes, max_attributes = self.compute_decision_tree(timer.sub_time(), authors, attributes,
                                                                         neighbors, options, tree_file)
        learning_time += timer.stop()

        timer.start("Converting decision tree to XADD")
        variables = [("f{}".format(i), int(0.8 * min_attributes[i]), int(1.2 * max_attributes[i]))
                     for i in range(self.attribute_count)]
        self.variables = variables
        row_variables, column_variables = self.get_row_column_variables()

        xadd = decision_tree_to_xadd(clf, row_variables + column_variables, discrete=discrete)
        for var in variables:
            xadd.pool.add_var(var[0], "int")
        matrix = Matrix(xadd, row_variables, column_variables).reduce()
        learning_time += timer.stop()
        self.learning_time = learning_time

        self._compute_pagerank(timer, matrix, variables, damping_factor, delta, iterations, diagram_file,
                               diagram_export_file)

        timer.start("Computing values for given authors")
        self.values = self.compute_values(attributes)
        self.grounding_time = timer.stop()

    def _compute_pagerank(self, timer, matrix, variables, damping_factor, delta, iterations, diagram_file=None,
                          diagram_export_file=None):
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
        self.converged = converged
        timer.log("Converged after {} iterations".format(iterations))
        self.pagerank_time = timer.stop()

    def compute_values(self, attributes):
        converged, attribute_count, variables = self.converged, self.attribute_count, self.variable_names
        assignments = []
        for i in range(len(attributes)):
            assignments.append({variables[j]: attributes[i][j] for j in range(attribute_count)})
        return mass_evaluate(converged.diagram, assignments)

    def get_decision_tree_accuracy(self, samples):
        pairs = set()
        examples = []

        def select():
            a1 = random.randint(0, len(self.authors) - 1)
            a2 = random.randint(0, len(self.authors) - 1)
            if a1 == a2 or (a1, a2) in pairs:
                return select()
            return a1, a2

        control = numpy.zeros(samples)
        for i in range(samples):
            a1, a2 = select()
            pairs.add((a1, a2))
            control[i] = 1 if a2 in self.neighbors[a1] else 0
            examples.append(self.attributes[a1] + self.attributes[a2])

        return self._compute_accuracy(examples, control)

    def get_balanced_tree_accuracy(self, samples):
        positive = negative = samples / 2

        pairs = set()
        examples = []
        control = numpy.zeros(samples)

        def select_negative():
            a1 = random.randint(0, len(self.authors) - 1)
            a2 = random.randint(0, len(self.authors) - 1)
            if a1 == a2 or (a1, a2) in pairs or a2 in self.neighbors[a1]:
                return select_negative()
            return a1, a2

        def select_positive():
            author1 = random.randint(0, len(self.authors) - 1)
            neighbors = self.neighbors[author1]
            if len(neighbors) == 0:
                return select_positive()
            author2 = neighbors[random.randint(0, len(neighbors) - 1)]
            if author1 == author2 or (author1, author2) in pairs:
                return select_positive()
            return author1, author2

        for i in range(negative):
            a1, a2 = select_negative()
            pairs.add((a1, a2))
            control[i] = 0
            examples.append(self.attributes[a1] + self.attributes[a2])

        for i in range(positive):
            a1, a2 = select_positive()
            pairs.add((a1, a2))
            control[negative + i] = 1
            examples.append(self.attributes[a1] + self.attributes[a2])

        return self._compute_accuracy(examples, control)

    def _compute_accuracy(self, examples, control):
        assert len(examples) == len(control)
        true_positive = 0
        positive = 0
        true_negative = 0
        negative = 0

        predicted = self.clf.predict(examples)

        for i in range(len(examples)):
            if control[i] == 1:
                if predicted[i] == 1:
                    true_positive += 1
                positive += 1
            else:
                if predicted[i] == 0:
                    true_negative += 1
                negative += 1

        # total = positive + negative
        return true_positive, true_negative, positive, negative

    def compute_decision_tree_test_accuracy(self, data_set, test_indices, sample_count):
        """
        Computes the accuracy of the decision tree on the test entries
        :type data_set: DataSet
        :type test_indices: int[]
        :type sample_count: int
        """
        test_index_set = set(test_indices)
        positive_candidates = set()
        for i in test_indices:
            for j in data_set.neighbors[i]:
                if j in test_index_set:
                    positive_candidates.add((i, j))
        if len(positive_candidates) <= sample_count:
            sample_count = len(positive_candidates)
            positive_examples = list(positive_candidates)
        else:
            positive_examples = random.sample(positive_candidates, sample_count)
        negative_candidates = set()
        while len(negative_candidates) < sample_count:
            a1 = random.choice(test_indices)
            a2 = random.choice(test_indices)
            key = (a1, a2)
            if a1 != a2 and key not in negative_candidates and key not in positive_candidates:
                negative_candidates.add(key)

        sum_papers = [data_set.sum_papers[i] for i in test_indices]
        sum_neighbors = [len(data_set.neighbors[i]) for i in test_indices]
        median_years = [data_set.median_years[i] for i in test_indices]
        attributes = dict(zip(test_indices, zip(sum_papers, sum_neighbors, median_years)))

        examples = positive_examples + list(negative_candidates)
        examples = [attributes[a1] + attributes[a2] for a1, a2 in examples]
        control = numpy.zeros(2 * sample_count)
        control[0:sample_count - 1] = 1
        return self._compute_accuracy(examples, control)


def count_links(neighbors):
    count = 0
    for i in range(len(neighbors)):
        count += len(neighbors[i])
    return count


def calculate_ground_pagerank(timer, authors, neighbors, damping_factor, delta, iterations):
    timer.start("Counting links")
    count = 0
    n = len(authors)
    for i in range(n):
        for _ in neighbors[i]:
            count += 1

    timer.log("{} links, density: {}".format(count, count / float(n * n)))

    timer.start("Creating sparse adjacency matrix")
    row = numpy.zeros(count)
    col = numpy.zeros(count)
    data = numpy.zeros(count)
    index = 0
    for i in range(n):
        for j in neighbors[i]:
            row[index] = i
            col[index] = j
            data[index] = 1
            # adjacency_matrix[i, j] = 1
            index += 1

    from scipy.sparse import coo_matrix
    from sparse import simple
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(n, n), dtype=numpy.float)
    # timer.start("Computing sparse page-rank")
    # values_ground, iterations = simple(adjacency_matrix, damping_factor, delta, iterations, norm=1)

    from sparse_pagerank_networkx import pagerank_scipy
    timer.start("Computing reference page-rank (networkx)")
    values_ground2 = pagerank_scipy(adjacency_matrix, alpha=damping_factor, max_iter=iterations, tol=delta)

    timer.start("Comparing page-rank results ({} iterations)".format(iterations))
    # for i in range(len(values_ground)):
    #     if abs(values_ground[i] - values_ground2[i]) > 2 * 10 ** -8:
    #         print("Solutions not in line: {} (sparse) vs {} (networkx)".format(values_ground[i], values_ground2[i]))

    # from sparse_pagerank_networkx import pagerank_scipy
    # values_ground1 = pagerank_scipy(adjacency_matrix, alpha=0.85, tol=10 ** -3)
    # values_ground, _ = sparse_pagerank.pageRank(adjacency_matrix, s=0.85, maxerr=10 ** -3)
    # values_test, _ = test_pagerank.pagerank_ground(numpy.matrix(adjacency_matrix), 0.85, 100, 10 ** -3)
    # for i in range(len(authors)):
    #     if not abs(values_test[i] - values_ground1[i]) < 10 ** -3 or not abs(values_ground[i] - values_test[i])
    # < 10 ** -3:
    #         raise RuntimeError("Index {}: not equal: {} (csc) {} (networkx) {} (test)".format(i, values_ground[i],
    # values_ground1[i], values_test[i]))
    #     else:
    #         print("check")
    # exit()
    return values_ground2  # values_ground  # [v[0, 0] for v in values_test]


def export_ground_pagerank(values_ground, path):
    with open(path, "w") as f:
        for author_value in values_ground:
            print(author_value, file=f)


def import_ground_value(path):
    with open(path, "r") as f:
        values = []
        for line in f:
            values.append(float(line.rstrip("\n")))
        return values


def sort_authors(authors, values):
    combined = zip(authors, values)
    sorted_authors = sorted(combined, key=lambda t: t[0])
    sorted_both = sorted(sorted_authors, key=lambda t: t[1], reverse=True)
    return [name for name, _ in sorted_both]


def make_histogram(values):
    histogram = {}
    for val in values:
        if val not in histogram:
            histogram[val] = 0
        histogram[val] += 1
    return histogram


class CitationExperimentSetting(object):
    def __init__(self, size, copy_rate, damping_factor, tree_depth, iterations, leaf_cutoff_rate,
                 verification_iterations, folds, top_count):
        self.size = size
        self.copy_rate = copy_rate
        self.damping_factor = damping_factor
        self.tree_depth = tree_depth
        self.iterations = iterations
        self.leaf_cutoff_rate = leaf_cutoff_rate
        self.verification_iterations = verification_iterations
        self.folds = folds
        self.top_count = top_count

    def get_experiment(self):
        experiment = CitationExperiment()
        experiment.size = self.size
        experiment.copy_rate = self.copy_rate
        experiment.damping_factor = self.damping_factor
        experiment.tree_depth = self.tree_depth
        experiment.iterations = self.iterations
        experiment.leaf_cutoff_rate = self.leaf_cutoff_rate
        experiment.verification_iterations = self.verification_iterations
        experiment.folds = self.folds
        experiment.top_count = self.top_count
        return experiment


class CitationExperiment(object):
    def __init__(self):
        self.size = numpy.nan
        self.copy_rate = numpy.nan
        self.links = numpy.nan
        self.damping_factor = numpy.nan
        self.tree_depth = numpy.nan
        self.true_positive = numpy.nan
        self.positive = numpy.nan
        self.true_negative = numpy.nan
        self.negative = numpy.nan
        self.kendall_tau = numpy.nan
        self.kt_ground_verification = numpy.nan
        self.kt_lifted_verification = numpy.nan
        self.unseen_kendall_tau_lifted_ground = numpy.nan
        self.unseen_kt_ground_verification = numpy.nan
        self.unseen_kt_lifted_verification = numpy.nan
        self.constant_verification = numpy.nan
        self.random_verification = numpy.nan
        self.iterations = numpy.nan
        self.lifted_speed_learning = numpy.nan
        self.lifted_speed_pagerank = numpy.nan
        self.lifted_speed_grounding = numpy.nan
        self.lifted_speed = numpy.nan
        self.ground_speed = numpy.nan
        self.leaf_cutoff_rate = numpy.nan
        self.verification_iterations = numpy.nan
        self.folds = numpy.nan
        self.top_count = numpy.nan
        self.seen_top_inclusion_ground_lifted = numpy.nan
        self.seen_top_inclusion_ground_verification = numpy.nan
        self.seen_top_inclusion_lifted_verification = numpy.nan
        self.unseen_top_inclusion_ground_lifted = numpy.nan
        self.unseen_top_inclusion_lifted_verification = numpy.nan
        self.unseen_top_inclusion_ground_verification = numpy.nan
        self.unseen_top_inclusion_constant_verification = numpy.nan
        self.unseen_top_inclusion_random_verification = numpy.nan
        self.seen_top_kt_ground_lifted = numpy.nan
        self.seen_top_kt_ground_verification = numpy.nan
        self.seen_top_kt_lifted_verification = numpy.nan
        self.unseen_top_kt_ground_lifted = numpy.nan
        self.unseen_top_kt_ground_verification = numpy.nan
        self.unseen_top_kt_lifted_verification = numpy.nan
        self.unseen_top_kt_constant_verification = numpy.nan
        self.unseen_top_kt_random_verification = numpy.nan

    @property
    def density(self):
        return self.links / float(self.size ** 2)

    @property
    def accuracy_positive(self):
        return self.true_positive / float(self.positive)

    @property
    def accuracy_negative(self):
        return self.true_negative / float(self.negative)

    def __repr__(self):
        return repr(self.export_to_dict())

    def export_to_dict(self):
        return self.__dict__
        # return {
        #     "size": self.size,
        #     "copy_rate": self.copy_rate,
        #     "links": self.links,
        #     "damping_factor": self.damping_factor,
        #     "tree_depth": self.tree_depth,
        #     "true_positive": self.true_positive,
        #     "positive": self.positive,
        #     "true_negative": self.true_negative,
        #     "negative": self.negative,
        #     "kendall_tau": self.kendall_tau,
        #     "kt_ground_verification": self.kt_ground_verification,
        #     "kt_lifted_verification": self.kt_lifted_verification,
        #     "iterations": self.iterations,
        #     "lifted_speed_learning": self.lifted_speed_learning,
        #     "lifted_speed_pagerank": self.lifted_speed_pagerank,
        #     "lifted_speed_grounding": self.lifted_speed_grounding,
        #     "lifted_speed": self.lifted_speed,
        #     "ground_speed": self.ground_speed,
        #     "leaf_cutoff_rate": self.leaf_cutoff_rate,
        #     "verification_iterations": self.verification_iterations,
        #     "folds": self.folds,
        # }

    @staticmethod
    def import_from_dict(dictionary):
        experiment = CitationExperiment()
        for key, value in dictionary.items():
            if hasattr(experiment, key):
                setattr(experiment, key, float(value))
        return experiment
        # experiment.size = int(dictionary["size"])
        # experiment.copy_rate = float(dictionary["copy_rate"])
        # experiment.links = int(dictionary["links"])
        # experiment.damping_factor = float(dictionary["damping_factor"])
        # experiment.tree_depth = int(dictionary["tree_depth"])
        # experiment.true_positive = int(dictionary["true_positive"])
        # experiment.positive = int(dictionary["positive"])
        # experiment.true_negative = int(dictionary["true_negative"])
        # experiment.negative = int(dictionary["negative"])
        # experiment.kendall_tau = float(dictionary["kendall_tau"])
        # experiment.kt_ground_verification = float(dictionary["kt_ground_verification"])
        # experiment.kt_lifted_verification = float(dictionary["kt_lifted_verification"])
        # experiment.iterations = int(dictionary["iterations"])
        # experiment.iterations = int(dictionary["iterations"])
        # experiment.lifted_speed_learning = float(dictionary["lifted_speed_learning"])
        # experiment.lifted_speed_pagerank = float(dictionary["lifted_speed_pagerank"])
        # experiment.lifted_speed_grounding = float(dictionary["lifted_speed_grounding"])
        # experiment.lifted_speed = float(dictionary["lifted_speed"])
        # experiment.ground_speed = float(dictionary["ground_speed"])
        # experiment.leaf_cutoff_rate = float(dictionary["leaf_cutoff_rate"])
        # experiment.verification_iterations = int(dictionary["verification_iterations"])
        # experiment.folds = int(dictionary["folds"])
        # return experiment


class DataSet(object):
    def __init__(self, authors, neighbors, sum_papers, median_years, model_xadd=None):
        self.authors, self.neighbors, self.sum_papers, self.median_years = authors, neighbors, sum_papers, median_years
        self.model_xadd = model_xadd

    @property
    def author_count(self):
        return len(self.authors)

    @property
    def link_count(self):
        return count_links(self.neighbors)

    def get_random_subset(self, size):
        if size < self.author_count:
            indices = random.sample(range(self.author_count), size)
        else:
            indices = list(range(self.author_count))
        return self.get_subset(indices)

    def get_subset(self, indices):
        subset = compute_subset(indices, self.authors, self.neighbors, self.sum_papers, self.median_years)
        authors, neighbors, sum_papers, median_years = subset
        data_set = DataSet(authors, neighbors, sum_papers, median_years)
        return data_set

    def get_attributes(self, indices=None):
        def filter_entries(entries):
            if indices is None:
                return entries
            else:
                return [entries[i] for i in indices]

        sum_papers = filter_entries(self.sum_papers)
        sum_neighbors = map(lambda l: len(l), filter_entries(self.neighbors))
        median_years = filter_entries(self.median_years)
        attributes = zip(sum_papers, sum_neighbors, median_years)
        if indices is not None:
            attributes = dict(zip(indices, attributes))
        return attributes

    def copy_neighbors(self, copy_rate):
        neighbors = copy_neighbors(self.neighbors, copy_rate)
        return DataSet(self.authors, neighbors, self.sum_papers, self.median_years)

    def reload(self, timer, location, identifier):
        self.export_to_file(timer, location, identifier)
        return DataSet.import_from_disk(timer, location, identifier)

    def as_list(self):
        return [self.authors, self.neighbors, self.sum_papers, self.median_years]

    def get_buckets(self, folds):
        bucket_indices = get_bucket_indices(folds, self.author_count)
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
        names = ["authors", "coauthors", "median_years"]
        f = lambda t: self.get_file_name(location, t, identifier)
        authors_file, coauthors_file, median_years_file = map(f, names)

        timer.start("Exporting authors to {}".format(authors_file))
        with open(authors_file, "w") as f:
            for i in range(self.author_count):
                print("{}, {}".format(self.authors[i], self.sum_papers[i]), file=f)

        timer.start("Exporting median years to {}".format(median_years_file))
        with open(median_years_file, "w") as f:
            for i in range(self.author_count):
                print("{}, {}".format(self.authors[i], self.median_years[i]), file=f)

        timer.start("Exporting coauthors to {}".format(coauthors_file))
        with open(coauthors_file, "w") as f:
            # TODO Consider changing the format to [node: neighbor1, ..., neighborN]
            for i in range(self.author_count):
                author_neighbors = self.neighbors[i]
                for neighbor in author_neighbors:
                    a1, a2 = sorted((self.authors[i], self.authors[neighbor]))
                    print("{}, {}".format(a1, a2), file=f)

        timer.stop()

    @staticmethod
    def import_from_disk(timer, location, identifier=None):
        names = ["authors", "coauthors", "median_years"]
        f = lambda t: DataSet.get_file_name(location, t, identifier)
        authors_file, coauthors_file, median_years_file = map(f, names)

        timer.start("Importing authors from {}".format(authors_file))
        authors, lookup, sum_papers = import_authors(authors_file)  # TODO sum paper is over all
        timer.start("Importing median years from {}".format(median_years_file))
        median_years = import_median_years(authors, median_years_file)  # TODO median years is over all
        timer.start("Importing coauthors from {}".format(coauthors_file))
        neighbors = import_neighbors(authors, lookup, coauthors_file)
        timer.stop()
        return DataSet(authors, neighbors, sum_papers, median_years)


class ExperimentRunner(object):
    authors_root_file = "authors.txt"
    coauthors_root_file = "coauthors.txt"
    median_years_root_file = "median_years.txt"

    def __init__(self, directory="temp", input_files_directory=None):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.experiments = []
        self.timer = Timer()
        self._full_data_set = None
        self.input_files_directory = input_files_directory

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
            top_count):
        self.load_data_set()
        data_size = self.full_data_set.author_count
        if data_size < size:
            size = data_size

        # pool_file = "temp/pool_{}.txt".format(size)

        model_tree_file = "{}/tree.txt".format(self.input_files_directory)
        model_tree_export_file = "{}/model_tree.dot".format(self.directory)

        cache_authors = True
        cache_ground_values = False

        timer = Timer()

        setting = CitationExperimentSetting(size, copy_rate, damping_factor, tree_depth, iterations, leaf_cutoff_rate,
                                            100, folds, top_count)

        options = {"max_depth": tree_depth, "min_samples_leaf": int(size * leaf_cutoff_rate)}
        subset = self.full_data_set.get_random_subset(size).reload(timer, self.directory, size)

        if copy_rate > 0:
            timer.start("Copy co-authors (copy rate={})".format(copy_rate))
            subset = subset.copy_neighbors(copy_rate)

        timer.start("Computing buckets ({} folds)".format(setting.folds))
        buckets = subset.get_buckets(setting.folds)

        # Below is the iterative part (per bucket)

        experiments = []
        measures = [(
            KendallTau(),
            "kendall_tau",
            "kt_ground_verification",
            "kt_lifted_verification",
            "unseen_kendall_tau_lifted_ground",
            "unseen_kt_ground_verification",
            "unseen_kt_lifted_verification",
            "constant_verification",
            "random_verification",
        ), (
            TopInclusion(top_count),
            "seen_top_inclusion_ground_lifted",
            "seen_top_inclusion_ground_verification",
            "seen_top_inclusion_lifted_verification",
            "unseen_top_inclusion_ground_lifted",
            "unseen_top_inclusion_ground_verification",
            "unseen_top_inclusion_lifted_verification",
            "unseen_top_inclusion_constant_verification",
            "unseen_top_inclusion_random_verification",
        ), (
            TopKT(top_count),
            "seen_top_kt_ground_lifted",
            "seen_top_kt_ground_verification",
            "seen_top_kt_lifted_verification",
            "unseen_top_kt_ground_lifted",
            "unseen_top_kt_ground_verification",
            "unseen_top_kt_lifted_verification",
            "unseen_top_kt_constant_verification",
            "unseen_top_kt_random_verification",
        )]

        for i in range(len(buckets)):
            tree_file = "{}/tree_{}_{}.dot".format(self.directory, size, i)
            diagram_file = "{}/diagram_{}_{}".format(self.directory, size, i)
            converged_file = "{}/converged_{}_{}".format(self.directory, size, i)
            values_ground_same_file = "{}/ground_value_same_{}_{}.txt".format(self.directory, size, i)
            values_ground_file = "{}/ground_value_{}_{}.txt".format(self.directory, size, i)
            values_full_ground_same_file = "{}/full_ground_value_same_{}_{}.txt".format(self.directory, size, i)
            values_full_ground_file = "{}/full_ground_value_{}_{}.txt".format(self.directory, size, i)

            training_indices, testing_indices = buckets[i]
            training_set = subset.get_subset(training_indices)

            experiment = setting.get_experiment()

            timer.start("Counting links")
            experiment.links = subset.link_count
            timer.stop()

            authors, neighbors, sum_papers, median_years = training_set.as_list()

            timer.start("Computing lifted values")
            task = AuthorPagerank(authors, neighbors, sum_papers, median_years)
            task.compute_pagerank(timer.sub_time(), damping_factor=damping_factor, delta=delta, iterations=iterations,
                                  tree_file=tree_file, diagram_file=diagram_file, diagram_export_file=None,
                                  discrete=discrete, options=options)
            values_lifted = task.values
            experiment.lifted_speed = timer.stop()
            experiment.lifted_speed_learning = task.learning_time
            experiment.lifted_speed_pagerank = task.pagerank_time
            experiment.lifted_speed_grounding = task.grounding_time

            timer.start("Computing decision tree accuracy")
            # true_positive, true_negative, positive, negative = task.get_balanced_tree_accuracy(100000)
            results = task.compute_decision_tree_test_accuracy(subset, testing_indices, 100000 / 2)
            true_positive, true_negative, positive, negative = results
            timer.log("TP = {}, TN = {}, P = {}, N = {}".format(true_positive, true_negative, positive, negative))
            experiment.true_positive = true_positive
            experiment.positive = positive
            experiment.true_negative = true_negative
            experiment.negative = negative

            timer.start("Exporting converged diagram to {}".format(converged_file))
            lifted_result = task.converged
            lifted_result.export(converged_file)

            if not cache_ground_values or not os.path.isfile(values_ground_same_file) \
                    or not os.path.isfile(values_ground_file):
                timer.start("Calculating ground pagerank (same number of iterations {})".format(iterations))
                ground_pagerank_same = calculate_ground_pagerank(timer.sub_time(), authors, neighbors,
                                                                 damping_factor=damping_factor,
                                                                 delta=delta, iterations=iterations)
                experiment.ground_speed = timer.stop()

                timer.start("Exporting ground pagerank to {}".format(values_ground_same_file))
                export_ground_pagerank(ground_pagerank_same, values_ground_same_file)

                # Verification
                verification = experiment.verification_iterations
                timer.start("Calculating ground pagerank (verification with {} iterations)".format(verification))
                ground_pagerank = calculate_ground_pagerank(timer.sub_time(), authors, neighbors,
                                                            damping_factor=damping_factor,
                                                            delta=delta, iterations=verification)

                timer.start("Exporting verification ground pagerank to {}".format(values_ground_file))
                export_ground_pagerank(ground_pagerank, values_ground_file)

                # Compute PageRank for all examples
                timer.start("Calculating full ground pagerank (same number of iterations {})".format(iterations))
                full_ground_pagerank_same = \
                    calculate_ground_pagerank(timer.sub_time(), subset.authors, subset.neighbors,
                                              damping_factor=damping_factor, delta=delta, iterations=iterations)
                # experiment.ground_speed = timer.stop()

                timer.start("Exporting ground pagerank to {}".format(values_full_ground_same_file))
                export_ground_pagerank(full_ground_pagerank_same, values_full_ground_same_file)

                verification = experiment.verification_iterations
                if model_tree_file is None:
                    timer.start("Calculating full ground pagerank (verification with {} iterations)".format(verification))
                    full_ground_pagerank = \
                        calculate_ground_pagerank(timer.sub_time(), subset.authors, subset.neighbors,
                                                  damping_factor=damping_factor, delta=delta, iterations=verification)
                else:
                    pool = Pool()
                    timer.start("Calculating lifted pagerank (verification with {} iterations)".format(iterations))
                    model_task = AuthorPagerank(None, None, None, None)
                    model_task.variables = task.variables  # TODO
                    row_variables, col_variables = model_task.get_row_column_variables()
                    for var in row_variables + col_variables:
                        pool.add_var(var[0], "int")
                    for var in model_task.variables:
                        pool.add_var(var[0], "int")
                    model_trees = read_xadds(model_tree_file, pool)
                    model_tree = model_trees[0]
                    matrix = Matrix(model_tree, row_variables, col_variables)
                    model_task._compute_pagerank(timer, matrix, model_task.variables, damping_factor, delta, iterations)

                    full_ground_pagerank = model_task.compute_values(subset.get_attributes())
                    timer.start("Exporting model tree")
                    matrix.export(model_tree_export_file)

                timer.start("Exporting verification ground pagerank to {}".format(values_full_ground_file))
                export_ground_pagerank(full_ground_pagerank, values_full_ground_file)

            timer.start("Importing ground pagerank from {}".format(values_ground_same_file))
            values_ground = import_ground_value(values_ground_same_file)

            timer.start("Importing ground verification pagerank from {}".format(values_ground_file))
            values_verification = import_ground_value(values_ground_file)

            # Compare PageRank for unseen examples
            timer.start("Importing full ground pagerank from {}".format(values_full_ground_same_file))
            values_full_ground = import_ground_value(values_full_ground_same_file)
            values_full_ground = list(values_full_ground[i] for i in testing_indices)

            timer.start("Importing full ground verification pagerank from {}".format(values_full_ground_file))
            values_full_verification = import_ground_value(values_full_ground_file)
            values_full_verification = list(values_full_verification[i] for i in testing_indices)

            testing_attributes = subset.get_attributes(testing_indices)
            testing_attributes = list(testing_attributes[i] for i in testing_indices)
            values_full_lifted = task.compute_values(testing_attributes)

            values_constant = numpy.zeros(len(values_full_verification))
            values_constant[::2] = 1
            values_random = list(random.random() for _ in range(len(values_full_verification)))

            for measure in measures:
                evaluation = measure[0]
                assert isinstance(evaluation, EvaluationMeasure)

                # Seen (reconstructive)
                seen_ground = ("ground", values_ground)
                seen_lifted = ("lifted", values_lifted)
                seen_verification = ("verification", values_verification)

                setattr(experiment, measure[1], evaluation.evaluate(timer, seen_ground, seen_lifted))
                setattr(experiment, measure[2], evaluation.evaluate(timer, seen_ground, seen_verification))
                setattr(experiment, measure[3], evaluation.evaluate(timer, seen_lifted, seen_verification))

                # Unseen (predictive)
                unseen_ground = ("unseen ground", values_full_ground)
                unseen_lifted = ("unseen lifted", values_full_lifted)
                unseen_verification = ("unseen verification", values_full_verification)

                setattr(experiment, measure[4], evaluation.evaluate(timer, unseen_ground, unseen_lifted))
                setattr(experiment, measure[5], evaluation.evaluate(timer, unseen_ground, unseen_verification))
                setattr(experiment, measure[6], evaluation.evaluate(timer, unseen_lifted, unseen_verification))

                # Baseline pagerank
                unseen_constant = ("unseen constant (2)", values_constant)
                unseen_random = ("unseen random", values_random)

                setattr(experiment, measure[7], evaluation.evaluate(timer, unseen_constant, unseen_verification))
                setattr(experiment, measure[8], evaluation.evaluate(timer, unseen_random, unseen_verification))

            timer.start("Calculating maximum")
            timer.log(find_optimal_conditions(lifted_result.diagram, task.variables))
            timer.stop()

            experiments.append(experiment)

        # histogram_lifted = make_histogram(values_lifted)
        # histogram_ground = make_histogram(values_ground)

        # print(histogram_lifted)
        # print(histogram_ground)

        self.experiments.append(experiments)
        return experiments

    def reset(self):
        self.experiments = []

    def export_experiments(self):
        output_file = "{}/output_{}.txt".format(self.directory, time.strftime("%Y%m%d_%H%M%S"))
        with open(output_file, "w") as stream:
            stream.write(json.dumps([map(lambda e: e.export_to_dict(), experiment) for experiment in self.experiments]))

    def import_experiments(self, output_file):
        path = "{}/{}".format(self.directory, output_file)
        with open(path) as stream:
            experiment_list = json.load(stream)
            self.experiments = list()
            for experiment in experiment_list:
                if isinstance(experiment, list):
                    self.experiments.append(map(CitationExperiment.import_from_dict, experiment))
                else:
                    self.experiments.append([CitationExperiment.import_from_dict(experiment)])

    @staticmethod
    def load_experiments(directory, output_file):
        runner = ExperimentRunner(directory)
        runner.import_experiments(output_file)
        return runner
