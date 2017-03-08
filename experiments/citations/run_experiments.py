from __future__ import print_function

import time

from experiments.citations.citations import ExperimentRunner, CitationExperiment
from experiments.citations.print_experiments import print_experiments


def main():
    size = 500000
    delta = 0
    iterations = 60
    damping_factor = 0.85
    copy_rate = 0
    discrete = True
    tree_depth = 5
    leaf_cutoff_rate = 0.01  # 0.001

    import uuid
    time_id = "{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), str(uuid.uuid4()))
    print("Running experiments with ID {}".format(time_id))
    directory = "temp_{}".format(time_id)
    runner = ExperimentRunner(directory)

    # for damping_factor in [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
    # for damping_factor in [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
    # for copy_rate in numpy.linspace(0, 1, 11):
    # for leaf_cutoff_rate in numpy.linspace(0.0001, 0.1001, 21):
    # for size in range(100000, 1100000, 100000):
    # for leaf_cutoff_rate in numpy.linspace(0.001, 0.1001, 21):
    # for damping_factor in numpy.linspace(0, 1, 21):
    for iterations in range(10, 70, 10):
        runner.run(size, delta, iterations, damping_factor, copy_rate, discrete, tree_depth, leaf_cutoff_rate)

    runner.export_experiments()

    print()
    print_experiments(runner.experiments)


if __name__ == "__main__":
    main()

