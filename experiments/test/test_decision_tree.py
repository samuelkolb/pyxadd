import unittest

from experiments.link_prediction import *
from pyxadd.matrix.matrix import assignments


def get_example1():
    n = 5
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    adjacency = edges_to_adjacency(edges, n)
    # sum papers, sum neighbors
    attribute_papers = [1, 2, 4, 2, 1]
    attribute_neighbors = [sum(adjacency[i]) for i in range(n)]

    attributes = zip(attribute_papers, attribute_neighbors)
    variables = [("r_f0", 0, 10), ("r_f1", 0, 10), ("c_f0", 0, 10), ("c_f1", 0, 10)]
    return adjacency, attributes, variables


class TestDecisionTree(unittest.TestCase):
    def test_example1(self):
        adjacency, attributes, variables = get_example1()
        examples, labels = graph_to_training_examples(adjacency, attributes)
        classifier = learn_decision_tree(examples, labels)
        diagram = decision_tree_to_xadd(classifier, variables)
        self.compare_decision_tree(variables, classifier, diagram)

    def test_run_example1(self):
        adjacency, attributes, variables = get_example1()
        examples, labels = graph_to_training_examples(adjacency, attributes)
        classifier = learn_decision_tree(examples, labels)
        diagram = decision_tree_to_xadd(classifier, variables)

        for i in range(len(attributes)):
            row = []
            for j in range(len(attributes)):
                feature_names = ["r_f0", "r_f1", "c_f0", "c_f1"]
                features = attributes[i] + attributes[j]
                row.append(diagram.evaluate(dict(zip(feature_names, features))))
            print(row)

        matrix = Matrix(diagram, variables[0:len(variables) / 2], variables[len(variables) / 2:])
        diagram.pool.add_var("f0", "int")
        diagram.pool.add_var("f1", "int")
        # TODO extract variables
        attribute_variables = [("f0", 0, 10), ("f1", 0, 10)]
        ranked, _ = pagerank.pagerank(matrix, 0.85, attribute_variables)
        ranked.print_ground()
        ranked.export("../visual/link_prediction/ranked")
        print(attributes)
        values = []
        for i in range(len(attributes)):
            node_attributes = attributes[i]
            features = {attribute_variables[j][0]: node_attributes[j] for j in range(len(node_attributes))}
            values.append((i, ranked.evaluate(features)))
        ordering = [index for index, value in sorted(values, key=lambda t: t[1], reverse=True)]
        print(ordering)

        print(numpy.matrix(adjacency))
        from experiments.test import test_pagerank
        ranked_ground, _ = test_pagerank.pagerank_ground(numpy.matrix(adjacency), 0.85, 100, 10 ** -3)
        print(ranked_ground)
        values_ground = [(i, float(ranked_ground[i])) for i in range(len(ranked_ground))]
        ordering_ground = [index for index, value in sorted(values_ground, key=lambda t: t[1], reverse=True)]
        print(ordering_ground)

    def compare_decision_tree(self, variables, classifier, diagram):
        for assignment in assignments(variables):
            features = [list((assignment[t[0]] for t in variables))]
            tree_prediction = classifier.predict(features)[0]
            diagram_evaluation = diagram.evaluate(assignment)
            self.assertEqual(tree_prediction, diagram_evaluation)
