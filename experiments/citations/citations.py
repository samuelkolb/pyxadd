from __future__ import print_function

import random

import os
import numpy
import scipy.sparse as sparse
import scipy.stats as stats
import time

from experiments.link_prediction import learn_decision_tree, decision_tree_to_xadd, export_classifier
from experiments.pagerank import pagerank
from experiments.test import test_pagerank
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
    index_set = set(indices)
    subset_authors = [complete_authors[i] for i in indices]
    subset_lookup = {subset_authors[i]: i for i in range(len(subset_authors))}
    subset_sum_papers = [complete_sum_papers[i] for i in indices]
    subset_median_years = [complete_median_years[i] for i in indices]
    subset_neighbors = [
        [subset_lookup[complete_authors[n]] for n in author_neighbors if n in index_set]
        for author_neighbors in [complete_neighbors[i] for i in indices]
    ]

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
        self.authors = authors
        self.neighbors = neighbors
        self.sum_papers = sum_papers
        self.median_years = median_years
        self.attributes = None
        self.converged = None
        self.values = None
        self.variables = None
        self.clf = None

    @property
    def attribute_count(self):
        return 3

    @property
    def variable_names(self):
        return ["f{}".format(i) for i in range(self.attribute_count)]

    def compute_pagerank(self, timer, damping_factor, delta, iterations, tree_file=None, diagram_file=None,
                         diagram_export_file=None, discrete=True, options=None):
        if self.converged is not None:
            return

        authors, neighbors, sum_papers, median_years = self.authors, self.neighbors, self.sum_papers, self.median_years

        timer.start("Computing attributes")
        sum_neighbors = []
        for i in range(len(authors)):
            sum_neighbors.append(len(neighbors[i]))

        attributes = zip(sum_papers, sum_neighbors, self.median_years)
        self.attributes = attributes
        # print(attributes)

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

        variables = [("f{}".format(i), min_attributes[i], int(1.2 * max_attributes[i])) for i in range(attribute_count)]
        self.variables = variables
        diagram_variables = []
        for prefix in ("r", "c"):
            diagram_variables.append([("{}_{}".format(prefix, name), lb, ub) for name, lb, ub in variables])
        row_variables, column_variables = diagram_variables

        examples = [attributes[i] + attributes[j] for i, j in examples]
        # print("\n".join(list(", ".join((str(e) for e in examples[i])) + ": " + str(labels[i]) for i in range(len(examples)))))

        timer.start("Learning decision tree")
        clf = learn_decision_tree(examples, labels, options)
        self.clf = clf

        if tree_file is not None:
            timer.start("Exporting decision tree to {}".format(tree_file))
            export_classifier(clf, tree_file)

        timer.start("Converting decision tree to XADD")
        xadd = decision_tree_to_xadd(clf, row_variables + column_variables, discrete=discrete)
        for var in variables:
            xadd.pool.add_var(var[0], "int")
        matrix = Matrix(xadd, row_variables, column_variables).reduce()

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

        timer.start("Computing values for given authors")
        self.values = self.compute_values(authors, attributes)

    def compute_values(self, authors, attributes):
        converged, attribute_count, variables = self.converged, self.attribute_count, self.variable_names
        assignments = []
        for i in range(len(authors)):
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
    #     if not abs(values_test[i] - values_ground1[i]) < 10 ** -3 or not abs(values_ground[i] - values_test[i]) < 10 ** -3:
    #         raise RuntimeError("Index {}: not equal: {} (csc) {} (networkx) {} (test)".format(i, values_ground[i], values_ground1[i], values_test[i]))
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


class CitationExperiment(object):
    def __init__(self):
        self.size = None
        self.copy_rate = None
        self.links = None
        self.damping_factor = None
        self.tree_depth = None
        self.true_positive = None
        self.positive = None
        self.true_negative = None
        self.negative = None
        self.kendall_tau = None
        self.iterations = None
        self.lifted_speed = None
        self.ground_speed = None
        self.leaf_cutoff_rate = None

    @property
    def density(self):
        return self.links / float(self.size ** 2)

    @property
    def accuracy_positive(self):
        return self.true_positive / float(self.positive)

    @property
    def accuracy_negative(self):
        return self.true_negative / float(self.negative)


def main(size, delta, iterations, damping_factor, copy_rate, discrete, tree_depth=5, leaf_cutoff_rate=0.001):
    """

    :rtype: CitationExperiment
    """
    authors_root_file = "authors.txt"
    coauthors_root_file = "coauthors.txt"
    median_years_root_file = "median_years.txt"

    authors_file = "authors_{}.txt".format(size)
    coauthors_file = "coauthors_{}.txt".format(size)
    median_years_file = "median_years_{}.txt".format(size)

    tree_file = "tree_{}.dot".format(size)
    diagram_file = "diagram_{}".format(size)
    converged_file = "converged_{}".format(size)
    values_ground_file = "ground_value_{}.txt".format(size)
    # pool_file = "pool_{}.txt".format(size)

    cache_authors = True
    cache_ground_values = False

    timer = Timer()

    experiment = CitationExperiment()
    experiment.size = size
    experiment.iterations = iterations
    experiment.damping_factor = damping_factor
    experiment.copy_rate = copy_rate
    experiment.tree_depth = tree_depth
    experiment.leaf_cutoff_rate = leaf_cutoff_rate

    if not cache_authors or not os.path.isfile(authors_file) or not os.path.isfile(coauthors_file)\
            or not os.path.isfile(median_years_file):
        timer.start("Importing authors from {} to compute subset".format(authors_root_file))
        authors, lookup, sum_papers = import_authors(authors_root_file)
        timer.start("Importing median years from {} to compute subset".format(median_years_root_file))
        median_years = import_median_years(authors, median_years_root_file)
        timer.start("Importing coauthors from {} to compute subset".format(coauthors_root_file))
        neighbors = import_neighbors(authors, lookup, coauthors_root_file)
        timer.start("Exporting author and coauthor files to {} and {}".format(authors_file, coauthors_file))
        export_subset(size, authors, neighbors, sum_papers, median_years, authors_file, coauthors_file,
                      median_years_file)

    timer.start("Importing authors from {}".format(authors_file))
    authors, lookup, sum_papers = import_authors(authors_file)

    timer.start("Importing median years from {}".format(median_years_file))
    median_years = import_median_years(authors, median_years_file)

    timer.start("Importing coauthors from {}".format(coauthors_file))
    neighbors = import_neighbors(authors, lookup, coauthors_file)

    timer.start("Copy co-authors (copy rate={})".format(copy_rate))
    neighbors = copy_neighbors(neighbors, copy_rate)

    timer.start("Counting links")
    experiment.links = count_links(neighbors)

    timer.start("Computing lifted values")
    task = AuthorPagerank(authors, neighbors, sum_papers, median_years)
    options = {"max_depth": tree_depth, "min_samples_leaf": int(size * leaf_cutoff_rate)}
    task.compute_pagerank(timer.sub_time(), damping_factor=damping_factor, delta=delta, iterations=iterations,
                          tree_file=tree_file, diagram_file=diagram_file, diagram_export_file=None,
                          discrete=discrete, options=options)
    values_lifted = task.values
    experiment.lifted_speed = timer.stop()

    timer.start("Computing decision tree accuracy")
    true_positive, true_negative, positive, negative = task.get_balanced_tree_accuracy(100000)
    timer.log("TP = {}, TN = {}, P = {}, N = {}".format(true_positive, true_negative, positive, negative))
    experiment.true_positive = true_positive
    experiment.positive = positive
    experiment.true_negative = true_negative
    experiment.negative = negative

    timer.start("Exporting converged diagram to {}".format(converged_file))
    task.converged.export(converged_file)

    if not cache_ground_values or not os.path.isfile(values_ground_file):
        timer.start("Calculating ground pagerank")
        ground_pagerank = calculate_ground_pagerank(timer.sub_time(), authors, neighbors, damping_factor=damping_factor,
                                                    delta=delta, iterations=iterations)
        experiment.ground_speed = timer.stop()

        timer.start("Exporting ground pagerank to {}".format(values_ground_file))
        export_ground_pagerank(ground_pagerank, values_ground_file)

    timer.start("Importing ground pagerank from {}".format(values_ground_file))
    values_ground = import_ground_value(values_ground_file)

    timer.start("Calculating kendall tau correlation coefficient")
    tau, _ = stats.kendalltau(values_lifted, values_ground)
    timer.log("KT = {}".format(tau))
    timer.stop()
    experiment.kendall_tau = tau

    timer.start("Calculating maximum")
    timer.log(find_optimal_conditions(task.converged.diagram, task.variables))
    timer.stop()

    # histogram_lifted = make_histogram(values_lifted)
    # histogram_ground = make_histogram(values_ground)

    # print(histogram_lifted)
    # print(histogram_ground)

    return experiment
