from __future__ import print_function
import scipy.stats as stats
import heapq


class EvaluationMeasure(object):
    def evaluate(self, timer, values1, values2):
        """
        Calculates the kendall tau measure
        :type timer: pyxadd.timer.Timer
        :type values1: Tuple[str, list]
        :type values2: Tuple[str, list]
        """
        raise NotImplementedError()


class KendallTau(EvaluationMeasure):
    def evaluate(self, timer, values1, values2):
        """
        Calculates the kendall tau measure
        :type timer: pyxadd.timer.Timer
        :type values1: Tuple[str, list]
        :type values2: Tuple[str, list]
        """
        timer.start("Calculating kendall tau correlation coefficient")
        tau, _ = stats.kendalltau(values1, values2)
        timer.log("KT = {}".format(tau))
        timer.stop()
        return tau


class TopInclusion(EvaluationMeasure):
    def __init__(self, top_count):
        self.top_count = top_count

    def evaluate(self, timer, values1, values2):
        """
        Calculates the kendall tau measure
        :type timer: pyxadd.timer.Timer
        :type values1: Tuple[str, list]
        :type values2: Tuple[str, list]
        """
        timer.start("Find top {} values".format(self.top_count))
        set1 = set(heapq.nlargest(self.top_count, range(len(values1)), key=lambda i: values1[i]))
        set2 = set(heapq.nlargest(self.top_count, range(len(values2)), key=lambda i: values2[i]))
        intersection = set1 & set2
        timer.log(str(set1))
        timer.log(str(set2))
        timer.log(str(intersection))
        similarity = float(len(intersection)) / self.top_count
        timer.log("Top inclusion = {}".format(similarity))
        timer.stop()
        return similarity


class TopKT(EvaluationMeasure):
    def __init__(self, top_count):
        self.top_count = top_count

    def evaluate(self, timer, values1, values2):
        """
        Calculates the kendall tau measure
        :type timer: pyxadd.timer.Timer
        :type values1: Tuple[str, list]
        :type values2: Tuple[str, list]
        """
        timer.start("Find top {} values".format(self.top_count))
        top1 = list(heapq.nlargest(self.top_count, values1))
        top2 = list(heapq.nlargest(self.top_count, values2))
        tau, _ = stats.kendalltau(top1, top2)
        timer.log("Top KT = {}".format(tau))
        timer.stop()
        return tau
