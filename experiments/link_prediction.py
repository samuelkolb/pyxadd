from __future__ import print_function

import os

import numpy
from pyxadd.view import export
from sklearn import tree

# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


from experiments import pagerank
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix
from tests.export import Exporter


def build_block_diagram():
    pool = Pool()
    build = Builder(pool)
    size = 20
    variables = [("i", 1, size)]
    for var in variables:
        name = var[0]
        build.ints("r_{}".format(name), "c_{}".format(name), name)
    limits = build.limit("r_i", 1, size) & build.limit("c_i", 1, size)
    block1 = (build.limit("r_i", 1, size / 2) & build.limit("c_i", 1, size / 2))
    block2 = (build.limit("r_i", size * 3 / 4 + 1, size) & build.limit("c_i", 1, size / 2))
    block3 = (build.limit("r_i", size / 4 + 1, size * 3 / 4) & build.limit("c_i", size / 4 + 1, size))
    diagram = limits & (block1 | block2 | block3)
    row_variables = [("r_" + name, lb, ub) for name, lb, ub in variables]
    column_variables = [("c_" + name, lb, ub) for name, lb, ub in variables]
    matrix = Matrix(diagram, row_variables, column_variables).reduce()

    # exporter = Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "link_prediction")
    # exporter.export(matrix, "matrix")
    # print("Matrix")
    # matrix.print_ground(row_limit=20, column_limit=20)

    projected = matrix.project(False)
    # print("Projected")
    # projected.print_ground(row_limit=20, column_limit=20)

    inverted = projected.transform_leaves(lambda terminal, d: d.pool.terminal(1.0 / terminal.expression))
    # inverted.print_ground(row_limit=20, column_limit=20)

    stochastic = matrix.element_product(inverted)
    # print("Stochastic")
    # stochastic.print_ground(row_limit=20, column_limit=20)

    return stochastic, variables


def learn_decision_tree(examples, labels, options=None):
    if options is None:
        options = {}
    clf = tree.DecisionTreeClassifier(**options)
    clf.fit(examples, labels)
    return clf


def decision_tree_to_xadd(pool, classifier, variables, discrete=True):
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold
    value = classifier.tree_.value

    build = Builder(pool)
    for t in variables:
        build.ints(str(t[0]))
    diagram = build.exp(0)
    stack = [(0, [])]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, path = stack.pop()

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            operator = "<=" if feature[node_id] >= 0 else ">"
            threshold_value = int(threshold[node_id])
            left_test = build.test(variables[feature[node_id]][0], "<=", threshold_value)
            right_test = build.test(variables[feature[node_id]][0], ">", threshold_value)
            # print("DECI", node_id, left_test.root_node.test, right_test.root_node.test)
            stack.append((children_left[node_id], path + [left_test]))
            stack.append((children_right[node_id], path + [right_test]))
        else:
            # print("LEAF", node_id, [d.root_node.test for d in path])
            distribution = value[node_id][0, :]
            # path_diagram = build.ite(build.test("i", "<=", 0), distribution[0], distribution[1])
            if discrete:
                constant = 1 if numpy.argmax(distribution) == 1 else 0
            else:
                constant = float(distribution[1]) / (distribution[0] + distribution[1])
            path_diagram = build.exp(constant)
            for test in path:
                path_diagram *= test
            # export(path_diagram, "visual/link_prediction/node{}.dot".format(node_id))
            diagram += path_diagram
    for v, lb, ub in variables:
        diagram *= build.limit(v, lb, ub)
    return diagram


def edges_to_adjacency(edges, n):
    return [[1 if tuple(sorted((row, col))) in edges else 0 for col in range(n)] for row in range(n)]


def graph_to_training_examples(adjacency, attributes):
    examples = []
    labels = []
    for i1 in range(len(attributes)):
        for i2 in range(len(attributes)):
            if i1 != i2:
                examples.append(attributes[i1] + attributes[i2])
                labels.append(adjacency[i1][i2])
    return examples, labels


def export_classifier(classifier, path="visual/link_prediction/tree.dot"):
    with open(path, 'w') as f:
        tree.export_graphviz(classifier, out_file=f)
