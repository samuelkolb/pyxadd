from __future__ import print_function

from os.path import dirname, basename

import numpy

from experiments.pr.pr_experiments import ExperimentRunner

attributes = [
    # Settings
    lambda e: e.size,
    lambda e: e.copy_rate,
    lambda e: e.links,
    lambda e: e.damping_factor,
    lambda e: e.iterations,
    lambda e: e.verification_iterations,
    lambda e: e.folds,
    lambda e: e.tree_depth,
    lambda e: e.leaf_cutoff_rate,
    lambda e: e.learning_sample_count,
    lambda e: e.all_positives,
    lambda e: e.true_positive_rate,
    lambda e: e.all_negatives,
    lambda e: e.true_negative_rate,
    lambda e: e.accuracy,
    lambda e: e.top_count,
    lambda e: e.kt_ground_true_training,
    lambda e: e.kt_lifted_true_training,
    lambda e: e.kt_ground_true_test,
    lambda e: e.kt_lifted_true_test,
    lambda e: e.kt_constant_true_test,
    lambda e: e.kt_random_true_test,
    lambda e: e.kt_top_ground_true_training,
    lambda e: e.kt_top_lifted_true_training,
    lambda e: e.kt_top_ground_true_test,
    lambda e: e.kt_top_lifted_true_test,
    lambda e: e.kt_top_constant_true_test,
    lambda e: e.kt_top_random_true_test,
    lambda e: e.top_inclusion_ground_true_training,
    lambda e: e.top_inclusion_lifted_true_training,
    lambda e: e.top_inclusion_ground_true_test,
    lambda e: e.top_inclusion_lifted_true_test,
    lambda e: e.top_inclusion_constant_true_test,
    lambda e: e.top_inclusion_random_true_test,
    lambda e: e.speed_learning_lifted,
    lambda e: e.speed_pagerank_lifted,
    lambda e: e.speed_grounding_lifted,
    lambda e: e.speed_lifted,
    lambda e: e.speed_ground,
]

names = [
    # Settings
    "Size",
    "Copy rate",
    "Number of edges",
    "Damping factor",
    "Iterations",
    "Verification iterations",

    "Folds",
    "Tree depth",
    "Leaf cutoff rate",
    "Learning sample count",

    "Positives count",
    "True positive rate",
    "Negatives count",
    "True negative rate",
    "Accuracy",

    "Top count",

    # Results
    "KT ground (training)",
    "KT lifted (training)",
    "KT ground (test)",
    "KT lifted (test)",
    "KT constant (test)",
    "KT random (test)",

    "KT-top ground (training)",
    "KT-top lifted (training)",
    "KT-top ground (test)",
    "KT-top lifted (test)",
    "KT-top constant (test)",
    "KT-top random (test)",

    "Top-inclusion ground (training)",
    "Top-inclusion lifted (training)",
    "Top-inclusion ground (test)",
    "Top-inclusion lifted (test)",
    "Top-inclusion constant (test)",
    "Top-inclusion random (test)",

    "Speed learning lifted",
    "Speed pagerank lifted",
    "Speed grounding lifted",
    "Speed all lifted",
    "Speed all ground",
]


def interleave(list1, list2):
    return [val for pair in zip(list1, list2) for val in pair]


def get_experiments_table(full_path, experiments):
    print_attributes = attributes
    titles = names
    titles = interleave(titles, ["std dev"] * len(titles))
    titles = ["Run ({})".format(full_path)] + titles
    header = "\t".join(titles)
    rows = [header]
    for i in range(len(experiments)):
        experiment = experiments[i]
        values = [numpy.average(list(t(e) for e in experiment)) for t in print_attributes]
        deviation = [numpy.std(list(t(e) for e in experiment)) for t in print_attributes]
        values = [i] + interleave(values, deviation)
        rows.append("\t".join(str(value) for value in values))
    return "\n".join(rows) + "\n"


def print_experiments(full_path, experiments, filename=None):
    if filename is None:
        print(get_experiments_table(full_path, experiments))
    else:
        with open(filename, "w") as stream:
            print(get_experiments_table(full_path, experiments), file=stream)

if __name__ == "__main__":
    # p = argparse.ArgumentParser()
    # p.add_argument("file")
    # arguments = vars(p.parse_args())
    full_path = "log/temp_20170812_020334_b63b27b7-d12a-46a9-9e56-820623de343b/output_20170812_020402.txt"  #arguments["file"]
    directory = dirname(full_path)
    output_file = basename(full_path)
    print(get_experiments_table(full_path, ExperimentRunner.import_experiments(directory, output_file)))
