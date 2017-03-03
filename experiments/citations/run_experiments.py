from __future__ import print_function

import numpy

from experiments.citations.citations import main as citation_main, CitationExperiment


def print_experiments(experiments):
    attributes = [
        ("Run", lambda experiment, i: i),
        ("Size", lambda experiment, i: experiment.size),
        ("Copy Rate", lambda experiment, i: experiment.copy_rate),
        ("Links", lambda experiment, i: experiment.links),
        ("Density", lambda experiment, i: experiment.density),
        ("Damping Factor", lambda experiment, i: experiment.damping_factor),
        ("Tree Depth", lambda experiment, i: experiment.tree_depth),
        ("Leaf Cutoff Rate", lambda experiment, i: experiment.leaf_cutoff_rate),
        ("Positive", lambda experiment, i: experiment.positive),
        ("Accuracy Positive", lambda experiment, i: experiment.accuracy_positive),
        ("Negative", lambda experiment, i: experiment.negative),
        ("Accuracy Negative", lambda experiment, i: experiment.accuracy_negative),
        ("Kendall Tau", lambda experiment, i: experiment.kendall_tau),
        ("Iterations", lambda experiment, i: experiment.iterations),
        ("Lifted Speed", lambda experiment, i: experiment.lifted_speed),
        ("Ground Speed", lambda experiment, i: experiment.ground_speed),
    ]

    print(*[t[0] for t in attributes], sep="\t")
    for i in range(len(experiments)):
        experiment = experiments[i]
        values = [t[1](experiment, i) for t in attributes]
        print(*values, sep="\t")


def main():
    size = 500000
    delta = 0
    iterations = 60
    damping_factor = 0.85
    copy_rate = 0
    discrete = True
    tree_depth = 5
    leaf_cutoff_rate = 0.01  # 0.001

    experiments = []

    # for damping_factor in [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
    # for damping_factor in [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
    # for copy_rate in numpy.linspace(0, 1, 11):
    # for leaf_cutoff_rate in numpy.linspace(0.0001, 0.1001, 21):
    # for size in range(100000, 1100000, 100000):
    # for leaf_cutoff_rate in numpy.linspace(0.001, 0.1001, 21):
    # for damping_factor in numpy.linspace(0, 1, 21):
    for iterations in range(10, 110, 10):
        experiments.append(citation_main(size, delta, iterations, damping_factor, copy_rate, discrete, tree_depth,
                                         leaf_cutoff_rate))

    print()
    print_experiments(experiments)



if __name__ == "__main__":
    main()
