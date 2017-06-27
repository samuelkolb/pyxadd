import os

from pyxadd import diagram as core, matrix_vector
from pyxadd.timer import Timer


class Variable(object):
    def __init__(self, name, v_type, lb=None, ub=None):
        assert isinstance(name, str)
        assert v_type in ["bool", "int", "real"]
        assert v_type != "bool" or (lb is None and ub is None)

        self.name = name
        self.v_type = v_type
        self.lb = lb
        self.ub = ub


class Benchmark(object):
    """
    diagram* methods return tuples of a diagram and its variables
    """

    @staticmethod
    def diagram1():
        """
        Load from data/test_evaluate_1.txt
        :rtype Tuple[Diagram, List[Variable]:
        """
        pool_file = os.path.dirname(os.path.realpath(__file__)) + "/data/test_evaluate_1.txt"
        root_id = 1663

        with open(pool_file, "r") as stream:
            json_input = stream.readline()

        exported_pool = core.Pool.from_json(json_input)
        diagram = exported_pool.diagram(root_id)
        var_list = [('r_f0', 0, 1658), ('r_f1', 0, 964), ('c_f0', 0, 1658), ('c_f1', 0, 964)]
        variables = [Variable(name, "int", lb, ub) for name, lb, ub in var_list]
        return diagram, variables


def repeat(times, benchmark_test, skip_first=True):
    """
    Repeat a test multiple times
    :param int times: How many times to repeat
    :param Callable[[], List[Tuple[str, float, float, *]]] benchmark_test:
    :param bool skip_first: if true, the first run is skipped
    :return List[Tuple[str, List[float], List[float], *]]: The aggregated results
    """
    aggregated = dict()
    threshold = 1 if skip_first is True else 0
    for i in range(times):
        results = benchmark_test()
        for method, build_time, exec_time, result in results:
            if i == threshold:
                aggregated[method] = ([build_time], [exec_time], result)
            elif i > threshold:
                aggregated[method][0].append(build_time)
                aggregated[method][1].append(exec_time)
    return [(key,) + aggregated[key] for key in sorted(aggregated.keys())]


def average(repeat_results):
    """
    Averages results from repeated runs (using repeat)
    :param List[Tuple[str, List[float], List[float], *]] repeat_results: The results from running repeat
    :return List[Tuple[str, float, float, *]]:
         A list of tuples containing each the diagram name, avg build time, avg execution time and operation result
    """
    from numpy import average as avg
    return list(map(lambda (m, b, e, r): (m, avg(b), avg(e), r), repeat_results))


def avg_std(repeat_results):
    """
    Averages results from repeated runs (using repeat)
    :param List[Tuple[str, List[float], List[float], *]] repeat_results: The results from running repeat
    :return List[Tuple[str, Tuple[float, float], Tuple[float, float], *]]:
         A list of tuples containing each the diagram name, avg build time, avg execution time and operation result
    """
    from numpy import average as avg, std
    return list(map(lambda (m, b, e, r): (m, (avg(b), std(b)), (avg(e), std(b)), r), repeat_results))


def run_all_diagrams(operation, verbose=False):
    """
    Runs an operation on all diagrams
    :param Callable[[Diagram, List[Variable], *] operation: The operation to run on diagrams
    :param bool verbose: If true results are printed out
    :return List[Tuple[str, float, float, *]]:
        A list of tuples containing each the diagram name, build time, execution time and operation result
    """
    diagram_filter = lambda method: callable(getattr(Benchmark, method)) and method.startswith("diagram")
    methods = list(sorted(filter(diagram_filter, dir(Benchmark))))

    results = []
    for method in methods:
        data = run_diagram(method, operation, verbose)
        results.append(data)
    return results


def run_diagram(method, operation, verbose):
    """
    Runs an operation on a diagram
    :param Callable[[Diagram, List[Variable], *] operation: The operation to run on the diagram
    :param bool verbose: If true results are printed out
    :return Tuple[str, float, float, *]:
        A tuple containing the diagram name, build time, execution time and operation result
    """
    timer = Timer(verbose=verbose)
    timer.start("Running {}".format(method))
    sub_time = timer.sub_time()
    sub_time.start("Building diagram")
    diagram, variables = getattr(Benchmark, method)()
    build_time = sub_time.stop()
    result = None
    sub_time.start("Running operation")
    if operation is not None:
        result = operation(diagram, variables)
    exec_time = sub_time.stop()
    if result is not None and verbose:
        timer.log("Operation result: {}".format(result))
    timer.stop()
    data = (method, build_time, exec_time, result)
    return data


if __name__ == "__main__":
    def eliminate_all(d, v):
        assert isinstance(d, core.Diagram)
        matrix_vector.sum_out(d.pool, d.root_id, [t.name for t in v])

    aggregated = average(repeat(1, lambda: run_all_diagrams(eliminate_all), skip_first=False))
    print("\n".join("Integration {} took {:.2f}s".format(data[0], data[2]) for data in aggregated))
