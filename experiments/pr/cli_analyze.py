from __future__ import print_function

import json
import os
from collections import defaultdict

from experiments.pr.pr_experiments import ExperimentRunner
from experiments.pr.print_experiments import get_experiments_table


def build_output_index(output_dir):
    experiment_dirs = next(os.walk(output_dir))[1]
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list())))
    for experiment_dir in experiment_dirs:
        id_file = "{}/{}/id.json".format(output_dir, experiment_dir)
        if os.path.exists(id_file):
            with open(id_file) as stream:
                data = json.load(stream)
            index[data["model"]][data["tree"]][data["data"]] += ["{}/{}".format(output_dir, experiment_dir)]
    return index


def build_model_index(models_dir):
    model_dirs = next(os.walk(models_dir))[1]
    index = defaultdict(lambda: defaultdict(lambda: list()))
    for model_dir in model_dirs:
        tree_list_file = "{}/{}/tree_list.json".format(models_dir, model_dir)
        if os.path.exists(tree_list_file):
            with open(tree_list_file) as stream:
                trees = [entry["id"] for entry in json.load(stream)]
            for tree in trees:
                tree_dir = "{}/{}/tree_{}".format(models_dir, model_dir, tree)
                data_list_file = "{}/data_list.json".format(tree_dir)
                with open(data_list_file) as stream:
                    index[model_dir][tree] = [entry["id"] for entry in json.load(stream)]
    return index


def print_output_index(index):
    for model in sorted(index):
        print("Model {}".format(model))
        for tree in sorted(index[model]):
            print("\tTree {}".format(tree))
            for data in sorted(index[model][tree]):
                print("\t\tData {}: {}".format(data, sorted(index[model][tree][data])))


def print_model_index(index):
    for model in sorted(index):
        print("Model {}".format(model))
        for tree in sorted(index[model]):
            print("\tTree {}".format(tree))
            print("\t\tData: {}".format(sorted(index[model][tree])))


def summarize(output_dir, analyze_dir, models=None):
    index = build_output_index(output_dir)
    for model in sorted(index) if models is None else models:
        model_dir = "{}/{}".format(analyze_dir, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Model {}".format(model))
        experiments = []
        for tree in sorted(index[model]):
            # tree_dir = "{}/tree{}".format(model_dir, tree)
            # if not os.path.exists(tree_dir):
            #     os.makedirs(tree_dir)
            print("\tTree {}".format(tree))
            output_files = []
            for data in sorted(index[model][tree]):
                # data_dir = "{}/data{}".format(tree_dir, data)
                # if not os.path.exists(data_dir):
                #     os.makedirs(data_dir)
                print("\t\tData {}: {}".format(data, sorted(index[model][tree][data])))
                for run_dir in sorted(index[model][tree][data]):
                    for dir_file in os.listdir(run_dir):
                        if os.path.isfile(os.path.join(run_dir, dir_file)) and "output" in dir_file:
                            output_files.append((run_dir, dir_file))
            print("\t\t{}".format(output_files))
            experiment = []
            for output_dir, output_file in output_files:
                experiment += ExperimentRunner.import_experiments(output_dir, output_file)[0]
            experiments += [experiment]
        with open("{}/overview.txt".format(model_dir), "w") as stream:
            table = get_experiments_table(model, experiments)
            print(table, file=stream)
        print(table + "\n")


def run(output_dir, model_dir, models, runs=1):
    output_index = build_output_index(output_dir)
    model_index = build_model_index(model_dir)
    for model in models:
        for tree in model_index[model]:
            for data in model_index[model][tree]:
                for i in range(runs - len(output_index[model][tree][data])):
                    from experiments.pr.run_experiments import main
                    main(output=output_dir, base_directory=model_dir, top_count=50, learning_sample_count=5000,
                         leaf_cutoff_rate=0.1, iterations=[40], model_name=model, tree_id=tree, data_id=data)


if __name__ == "__main__":
    output_dir = "./log"
    models_dir = "./models"
    analyze_dir = "./analyze"
    models = ["model4"]
    # run(output_dir, models_dir, models)
    summarize(output_dir, analyze_dir, models)
