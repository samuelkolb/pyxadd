from __future__ import print_function

import time

import sys

from experiments.citations.citations import ExperimentRunner, CitationExperiment
from experiments.citations.print_experiments import print_experiments


def main(output=".", size=None, delta=None, iterations=None, learning_sample_count=None, damping_factor=None,
         copy_rate=None, discrete=None, tree_depth=None, leaf_cutoff_rate=None, folds=None, top_count=None,
         input_files_directory=None):
    settings = dict()
    settings["size"] = 500000 if size is None else size
    settings["delta"] = 0 if delta is None else delta
    settings["iterations"] = 60 if iterations is None else iterations
    settings["damping_factor"] = 0.85 if damping_factor is None else damping_factor
    settings["copy_rate"] = 0 if copy_rate is None else copy_rate
    settings["discrete"] = True if discrete is None else discrete
    settings["tree_depth"] = 5 if tree_depth is None else tree_depth
    settings["leaf_cutoff_rate"] = 0.01 if leaf_cutoff_rate is None else leaf_cutoff_rate
    settings["folds"] = 5 if folds is None else folds
    settings["top_count"] = 50 if top_count is None else top_count
    settings["learning_sample_count"] = learning_sample_count

    variable = None
    for name, value in settings.items():
        if isinstance(value, list):
            if variable is not None:
                raise RuntimeError("Multiple range search currently not supported")
            variable = name

    import uuid
    time_id = "{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), str(uuid.uuid4()))
    print("Running experiments with ID {}".format(time_id))
    directory = "{}/temp_{}".format(output, time_id)
    runner = ExperimentRunner(directory, input_files_directory)

    # for damping_factor in [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
    # for damping_factor in [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
    # for copy_rate in numpy.linspace(0, 1, 11):
    # for leaf_cutoff_rate in numpy.linspace(0.0001, 0.1001, 21):
    # for size in range(100000, 1100000, 100000):
    # for leaf_cutoff_rate in numpy.linspace(0.001, 0.1001, 21):
    # for damping_factor in numpy.linspace(0, 1, 21):

    def run(values):
        runner.run(
            values["size"],
            values["delta"],
            values["iterations"],
            values["damping_factor"],
            values["copy_rate"],
            values["discrete"],
            values["tree_depth"],
            values["leaf_cutoff_rate"],
            values["folds"],
            values["top_count"],
            values["learning_sample_count"]
        )

    if variable is None:
        run(settings)
    else:
        values = settings
        for current_value in settings[variable]:
            values[variable] = current_value
            run(values)

    runner.export_experiments()

    print("\nExperiments for ID {}".format(time_id))
    print_experiments(runner.experiments)
    table_file = "{}/table.txt".format(directory)
    print_experiments(runner.experiments, filename=table_file)

if __name__ == "__main__":
    main(output="./log", input_files_directory="./models/data_uniform1", top_count=100, learning_sample_count=20000,
         #leaf_cutoff_rate=0.05, damping_factor=[0, 0.2, 0.4, 0.6, 0.8, 1])
    leaf_cutoff_rate=[0.1, 0.05, 0.01, 0.005, 0.001])
