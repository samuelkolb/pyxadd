import numpy


class ExpConfig(object):
    def __init__(self, size, copy_rate, damping_factor, tree_depth, iterations, leaf_cutoff_rate,
                 verification_iterations, folds, top_count, learning_sample_count):
        self.size = size
        self.copy_rate = copy_rate
        self.damping_factor = damping_factor
        self.tree_depth = tree_depth
        self.iterations = iterations
        self.leaf_cutoff_rate = leaf_cutoff_rate
        self.verification_iterations = verification_iterations
        self.folds = folds
        self.top_count = top_count
        if learning_sample_count is not None:
            self.learning_sample_count = learning_sample_count

    def get_experiment(self):
        experiment = ExpResults()
        experiment.size = self.size
        experiment.copy_rate = self.copy_rate
        experiment.damping_factor = self.damping_factor
        experiment.tree_depth = self.tree_depth
        experiment.iterations = self.iterations
        experiment.leaf_cutoff_rate = self.leaf_cutoff_rate
        experiment.verification_iterations = self.verification_iterations
        experiment.folds = self.folds
        experiment.top_count = self.top_count
        experiment.learning_sample_count = self.learning_sample_count
        return experiment


class ExpResults(object):
    def __init__(self):
        # Settings
        self.size = numpy.nan
        self.copy_rate = numpy.nan
        self.links = numpy.nan
        self.damping_factor = numpy.nan
        self.iterations = numpy.nan
        self.verification_iterations = numpy.nan

        self.folds = numpy.nan
        self.tree_depth = numpy.nan
        self.leaf_cutoff_rate = numpy.nan
        self.learning_sample_count = numpy.nan

        self.true_positive = numpy.nan
        self.false_positive = numpy.nan
        self.true_negative = numpy.nan
        self.false_negative = numpy.nan

        self.top_count = numpy.nan

        # Results
        self.kt_ground_true_training = numpy.nan
        self.kt_lifted_true_training = numpy.nan
        self.kt_ground_true_test = numpy.nan
        self.kt_lifted_true_test = numpy.nan
        self.kt_constant_true_test = numpy.nan
        self.kt_random_true_test = numpy.nan

        self.kt_top_ground_true_training = numpy.nan
        self.kt_top_lifted_true_training = numpy.nan
        self.kt_top_ground_true_test = numpy.nan
        self.kt_top_lifted_true_test = numpy.nan
        self.kt_top_constant_true_test = numpy.nan
        self.kt_top_random_true_test = numpy.nan

        self.top_inclusion_ground_true_training = numpy.nan
        self.top_inclusion_lifted_true_training = numpy.nan
        self.top_inclusion_ground_true_test = numpy.nan
        self.top_inclusion_lifted_true_test = numpy.nan
        self.top_inclusion_constant_true_test = numpy.nan
        self.top_inclusion_random_true_test = numpy.nan

        self.speed_learning_lifted = numpy.nan
        self.speed_pagerank_lifted = numpy.nan
        self.speed_grounding_lifted = numpy.nan
        self.speed_lifted = numpy.nan
        self.speed_ground = numpy.nan

    @property
    def density(self):
        return self.links / float(self.size ** 2)

    @property
    def all_positives(self):
        return self.true_positive + self.false_negative

    @property
    def all_negatives(self):
        return self.true_negative + self.false_positive

    @property
    def true_positive_rate(self):
        return self.true_positive / float(self.all_positives)

    @property
    def true_negative_rate(self):
        return self.true_negative / float(self.all_negatives)

    @property
    def accuracy(self):
        return (self.true_positive + self.true_negative) / float(self.all_positives + self.all_negatives)

    def __repr__(self):
        return repr(self.export_to_dict())

    def export_to_dict(self):
        return self.__dict__

    @staticmethod
    def import_from_dict(dictionary):
        experiment = ExpResults()
        for key, value in dictionary.items():
            if hasattr(experiment, key):
                setattr(experiment, key, float(value))
        return experiment

