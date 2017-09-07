from __future__ import print_function

import json
import random
import sys
import os
import subprocess

import numpy
from experiments.citations.parse import read_xadds
import pyxadd.diagram as core

try:
    input = raw_input
except NameError:
    pass


problog_dt_template = """
:- consult('{dt_problog}').
features({features}).
classes([0,1]).
maxdepth({max_depth}).
uniform(Low,High)::select_value(_,Low,High).
hidden_concept(Tree) :- previous(hidden_concept(Tree), generate_tree(Tree)).
output(Tree) :- hidden_concept(Tree).
query(output(_))."""


class ModelManager(object):
    def __init__(self, models_dir, model_name):
        self.models_dir = models_dir
        self.model_name = model_name

    @property
    def model_dir(self):
        return "{}/{}".format(self.models_dir, self.model_name)

    @property
    def config_filename(self):
        return "{}/config.json".format(self.model_dir)

    @property
    def tree_list_filename(self):
        return "{}/tree_list.json".format(self.model_dir)

    def create_tree(self, max_depth):
        with open(self.config_filename) as stream:
            model = json.load(stream)
        variables = model["variables"]
        features = [("r_{}".format(f), lb, ub) for f, lb, ub in variables] \
                   + [("c_{}".format(f), lb, ub) for f, lb, ub in variables]

        tree_problog_filename = "{}/tree_{}.pl".format(self.model_dir, max_depth)
        if not os.path.exists(tree_problog_filename):
            dt_problog = "{}/problog/decision_tree".format(os.path.dirname(os.path.realpath(__file__)))
            with open(tree_problog_filename, "w") as stream:
                string = problog_dt_template.format(dt_problog=dt_problog, features=str(features), max_depth=max_depth)
                print(string, file=stream)

        with open(self.tree_list_filename) as stream:
            tree_list = json.load(stream)
        tree_id = 0 if len(tree_list) == 0 else max(tree["id"] for tree in tree_list) + 1
        tree_dir = "{}/tree_{}".format(self.model_dir, tree_id)
        if not os.path.exists(tree_dir):
            os.makedirs(tree_dir)
        tree_file = "{}/tree.txt".format(tree_dir)
        subprocess.call(["problog", "sample", tree_problog_filename, "-N", "1", "-o", tree_file])
        tree_list.append({"id": tree_id, "max_depth": max_depth})
        with open(self.tree_list_filename, "w") as stream:
            json.dump(tree_list, stream)
        with open("{}/data_list.json".format(tree_dir), "w") as stream:
            json.dump([], stream)
        return tree_id

    def create_data_set(self, tree_id, node_count):
        with open(self.config_filename) as stream:
            model = json.load(stream)
        variables = model["variables"]
        features = [("r_{}".format(f), lb, ub) for f, lb, ub in variables]\
            + [("c_{}".format(f), lb, ub) for f, lb, ub in variables]

        data_list_filename = "{}/tree_{}/data_list.json".format(self.model_dir, tree_id)
        with open(data_list_filename) as stream:
            data_list = list(json.load(stream))
        data_id = 0 if len(data_list) == 0 else max(data["id"] for data in data_list) + 1
        data_dir = "{}/tree_{}/data_{}".format(self.model_dir, tree_id, data_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        attributes_filename = "{}/attributes.txt".format(data_dir)
        edges_filename = "{}/edges.txt".format(data_dir)

        attributes = numpy.zeros([node_count, len(variables)], "int")
        with open(attributes_filename, "w") as stream:
            for i in range(node_count):
                for j in range(len(variables)):
                    attributes[i, j] = random.randint(variables[j][1], variables[j][2])
                print(", ".join(str(e) for e in ([i] + list(attributes[i, :]))), file=stream)

        edges = []
        for i in range(node_count):
            edges.append([])
        xadd = read_xadds("{}/tree_{}/tree.txt".format(self.model_dir, tree_id), core.Pool())[0]
        for i in range(node_count):
            for j in range(node_count):
                assignment = dict(zip(map(lambda t: t[0], features), list(attributes[i, :]) + list(attributes[j, :])))
                result = xadd.evaluate(assignment)
                if result == 1:
                    edges[i].append(j)
        with open(edges_filename, "w") as stream:
            for i in range(node_count):
                print(", ".join(str(e) for e in edges[i]), file=stream)
        data_list.append({"id": data_id, "node_count": node_count})
        with open(data_list_filename, "w") as stream:
            json.dump(data_list, stream)
        return data_id


def main(models_dir, model_name):
    print("Make a new model, tree or data set? [model|tree|data]")
    action = input()

    manager = ModelManager(models_dir, model_name)

    if action == "model":
        if not os.path.exists(manager.model_dir):
            os.makedirs(manager.model_dir)

        print("How many features? [<int>]")
        features = int(input())

        print("Equal domain size? [yes|no]")
        equal_domains = input()
        if equal_domains == "yes":
            print("What is the domain size? [<int>]")
            domain_size = int(input())
            feature_domain_sizes = [domain_size] * features
        elif equal_domains == "no":
            feature_domain_sizes = []
            for i in range(features):
                print("What is the domain size of feature {} (f{})?".format(i + 1, i))
                feature_domain_sizes.append(int(input()))
        else:
            raise RuntimeError("Not a valid answer '{}', should be 'yes' or 'no'")
        variables = [("f{}".format(i), 1, ds) for i, ds in enumerate(feature_domain_sizes)]
        with open(manager.config_filename, "w") as stream:
            json.dump({"variables": variables}, stream)
        with open(manager.tree_list_filename, "w") as stream:
            json.dump([], stream)
    elif action == "tree":
        print("What is the maximum depth? [<int>]")
        max_depth = int(input())

        print("How many trees do you want to generate? [<int>]")
        tree_count = int(input())

        print("How many data sets do you want to generate per tree? [<int>]")
        data_count = int(input())
        node_count = None

        if data_count > 0:
            print("How many nodes should be generated in each data set? [<int]")
            node_count = int(input())

        for _i in range(tree_count):
            tree_id = manager.create_tree(max_depth)
            if data_count > 0:
                for _j in range(data_count):
                    manager.create_data_set(tree_id, node_count)

    elif action == "data":
        print("Which tree should the data be generated for? [<int>]")
        tree_id = int(input())

        print("How many nodes should be generated? [<int>]")
        node_count = int(input())

        print("How many data sets should be generated? [<int>]")
        data_count = int(input())

        for _ in range(data_count):
            manager.create_data_set(tree_id, node_count)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
