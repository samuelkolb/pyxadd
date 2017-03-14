from __future__ import print_function

import sympy
from collections import defaultdict

from pyxadd.diagram import Diagram, DefaultCache, Pool
from pyxadd.operation import Summation, Multiplication
from pyxadd.test import LinearTest
from pyxadd.variables import VariableFinder
from pyxadd.walk import DownUpWalker


class SummationCache(DefaultCache):

    name = "summation-cache"

    def __init__(self):
        self.lb = sympy.S("lb")
        self.ub = sympy.S("ub")

        def calculator(pool, var_node):
            """
            Symbolically sums the expression of the node.
            :return: a lambda expression that substitutes the given lower- and upper bound in the sum
            """
            variable, node_id = var_node
            expression = pool.get_node(node_id).expression
            variables = {str(v): v for v in expression.free_symbols}

            if len(variables) == 0:
                return lambda lb, ub: (ub - lb + 1) * float(expression)

            v = variables[variable] if variable in variables else sympy.S(variable)
            # TODO add caching again
            try:
                # expression = sympy.expand(expression)
                # print("Value at r=10 is {}".format(expression.subs({"r": 10})))
                result = sympy.Sum(expression, (v, self.lb, self.ub)).doit()
                # print("Symbolic sum of {} = {}".format(expression, result))
                return lambda lb, ub: result.subs({self.lb: lb, self.ub: ub})
            except sympy.BasePolynomialError as e:
                print("Problem trying to sum the expression {} for variable {}"
                      .format(expression, v))
                raise e

        super(SummationCache, self).__init__(calculator)

    @classmethod
    def initialize(cls, pool):
        if not pool.has_cache(cls.name):
            pool.add_cache(cls.name, cls())


def filter_bounds(bounds, lower):
    """
    Filters the given bounds to eliminate redundant ones
    :param List bounds: The bounds to filter
    :param bool lower: True iff the bounds are lower bounds, False otherwise
    :return List: The filtered list of bounds
    """
    collected_bounds = []

    for bound_1 in bounds:
        if len(collected_bounds) == 0:
            # No bounds so far
            collected_bounds = [bound_1]
        else:
            next_collected_bounds = []
            for bound_2 in collected_bounds:
                diff = bound_1 - bound_2
                if len(diff.free_symbols) == 0:
                    if lower:
                        next_collected_bounds.append(bound_1 if diff >= 0 else bound_2)
                    else:
                        next_collected_bounds.append(bound_1 if diff <= 0 else bound_2)
                else:
                    next_collected_bounds.append(bound_1)
                    next_collected_bounds.append(bound_2)
            collected_bounds = next_collected_bounds
    return collected_bounds


class SummationWalker(DownUpWalker):
    def __init__(self, diagram, variable):
        DownUpWalker.__init__(self, diagram)
        self.variable = str(variable)
        # The node cache keeps track of the updated bounds per test
        self.node_cache = dict()
        self.sum_cache = dict()
        SummationCache.initialize(diagram.pool)
        self.conflicts = set()
        self.revisit = defaultdict(lambda: 0)

    def visit_internal_down(self, internal_node, parent_message):
        operator = internal_node.test.operator.to_canonical()
        # expression = internal_node.test.expression

        # Initialize bounds
        if parent_message is not None:
            lb, ub, bounds = parent_message
        else:
            lb, ub, bounds = -float("inf"), float("inf"), []

        if operator.is_singular() and self.variable in operator.variables:
            # Test on exactly the given variable: update bounds for the two children (node will be collapsed)
            lb_t, ub_t = internal_node.test.update_bounds(self.variable, lb, ub, test=True)
            lb_f, ub_f = internal_node.test.update_bounds(self.variable, lb, ub, test=False)
            return (lb_t, ub_t, bounds), (lb_f, ub_f, bounds)

        elif len(operator.variables) > 1 and self.variable in operator.variables:
            # Test that includes the given variable and others: rewrite and pass both options (node will be collapsed)
            def to_exp(op):
                expression = sympy.sympify(op.rhs)
                for k, v in op.lhs.items():
                    if k != self.variable:
                        expression = -sympy.S(k) * v + expression
                return expression

            rewritten_positive = operator.times(1 / operator.coefficient(self.variable)).weak()
            exp_pos = to_exp(rewritten_positive)

            rewritten_negative = (~operator).times(1 / operator.coefficient(self.variable)).weak()
            exp_neg = to_exp(rewritten_negative)

            true_bound = (rewritten_positive.symbol, exp_pos)
            false_bound = (rewritten_negative.symbol, exp_neg)
            return (lb, ub, bounds + [true_bound]), (lb, ub, bounds + [false_bound])

        else:
            # Test that does not include the given variable (node test will be maintained)
            self.node_cache[internal_node.node_id] = internal_node.test
            return (lb, ub, bounds), (lb, ub, bounds)

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        pool = self._diagram.pool
        if internal_node.node_id in self.node_cache:
            # The node test is maintained and the two results become the new child nodes
            test_node = pool.bool_test(self.node_cache[internal_node.node_id])
            result = pool.apply(Summation,
                                pool.apply(Multiplication, test_node, true_result),
                                pool.apply(Multiplication, pool.invert(test_node), false_result))
            return result

        result = pool.apply(Summation, true_result, false_result)
        return result

    def visit_terminal(self, terminal_node, parent_message):
        if parent_message is None:
            return terminal_node.node_id

        pool = self._diagram.pool
        lb_natural, ub_natural, bounds = parent_message

        if terminal_node.expression == 0 or lb_natural > ub_natural:
            return pool.zero_id

        lower_bounds = []
        upper_bounds = []
        for bound in bounds:
            # [var] [operator] [expression]
            operator, expression = bound

            if operator == ">=":
                lower_bounds.append(expression)
            elif operator == "<=":
                upper_bounds.append(expression)
            else:
                raise RuntimeError("Cannot handle operator {}".format(operator))

        # print("Expression: {}".format(terminal_node.expression))

        # print("Lower bounds: {}".format(lower_bounds))
        # print("Upper bounds: {}".format(upper_bounds))

        lower_bounds = filter_bounds(lower_bounds, lower=True)
        upper_bounds = filter_bounds(upper_bounds, lower=False)

        lower_bounds.append(lb_natural)
        upper_bounds.append(ub_natural)

        # print("Lower bounds: {}".format(lower_bounds))
        # print("Upper bounds: {}".format(upper_bounds))

        # print()
        # from pyxadd.timer import Timer
        # timer = Timer()
        # timer.start("Building terminal {}".format(terminal_node.node_id))
        converted_terminal = self._build_terminal(terminal_node, 0, 1, lower_bounds, 0, 1, upper_bounds)
        # timer.stop()
        return converted_terminal

    def _build_terminal(self, terminal_node, lb_i, lb_c, lower_bounds, ub_i, ub_c, upper_bounds):
        self.revisit[terminal_node.node_id] += 1
        pool = self._diagram.pool
        assert isinstance(pool, Pool)

        # Check lower bounds
        if lb_c == len(lower_bounds):
            # Check upper bounds
            if ub_c == len(upper_bounds):
                lb = lower_bounds[lb_i]
                ub = upper_bounds[ub_i]
                import time
                # result = sympy.nsimplify(sympy.Sum(sympy.nsimplify(expression), (self.variable, lb, ub)).doit())
                node_id = terminal_node.node_id
                # hit = pool.is_cached(SummationCache.name, (self.variable, node_id))
                f = pool.get_cached(SummationCache.name, (self.variable, node_id))
                result = f(lb, ub)
                if result == sympy.nan:
                    raise RuntimeError("Result is nan: {} for lb={} and ub={}".format(terminal_node.expression, lb, ub))
                # print("Leaf sum:", ("cached" if hit else "not cached"), (lb, ub), result)
                # TODO: simplify result numerically??
                bound_integrity_check = LinearTest(lb, "<=", ub)
                if bound_integrity_check.operator.is_tautology():
                    return pool.terminal(result) if bound_integrity_check.evaluate({}) else pool.zero_id
                else:
                    node_id = pool.bool_test(bound_integrity_check)
                    return pool.apply(Multiplication, node_id, pool.terminal(result))
            else:
                # Add upper bound check
                test = LinearTest(upper_bounds[ub_i], "<=", upper_bounds[ub_c])
                child_true = self._build_terminal(terminal_node, lb_i, lb_c, lower_bounds, ub_i, ub_c + 1, upper_bounds)
                child_false = self._build_terminal(terminal_node, lb_i, lb_c, lower_bounds, ub_c, ub_c + 1, upper_bounds)

        # FIXME TRANSITIVITY
        else:
            # Add lower bound check
            test = LinearTest(lower_bounds[lb_i], ">=", lower_bounds[lb_c])
            child_true = self._build_terminal(terminal_node, lb_i, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)
            child_false = self._build_terminal(terminal_node, lb_c, lb_c + 1, lower_bounds, ub_i, ub_c, upper_bounds)

        if test.operator.is_tautology():
            return child_true if test.evaluate({}) else child_false

        test_node = pool.bool_test(test)
        return pool.apply(Summation,
                          pool.apply(Multiplication, test_node, child_true),
                          pool.apply(Multiplication, pool.invert(test_node), child_false))


def matrix_multiply(pool, root1, root2, variables):
    """
    :type pool: Pool
    :type root1: int
    :type root2: int
    :type variables: list
    """
    return sum_out(pool, pool.apply(Multiplication, root1, root2), variables)


def matrix_multiply_reduced(pool, root1, root2, variables, reducer=None, all_variables=None):
    """
    :type pool: Pool
    :type root1: int
    :type root2: int
    :type variables: list
    :type reducer: pyxadd.reduce.Reducer|None
    :type all_variables: list|None
    """
    multiplied = pool.apply(Multiplication, root1, root2)
    if reducer is not None:
        multiplied = reducer.reduce(multiplied, all_variables)
    return sum_out(pool, multiplied, variables, reducer, all_variables)


def sum_out(pool, root, variables, reducer=None, all_variables=None):
    variables = list(str(v) for v in variables)
    diagram = pool.diagram(root)
    result = diagram
    # from pyxadd.timer import Timer
    # timer = Timer()
    # timer.start("Summing out")
    # per_var = timer.sub_time()
    for var in variables:
        # per_var.start("Summing out {}".format(var))
        walker = SummationWalker(result, var)
        result_id = walker.walk()
        if reducer is not None:
            result_id = reducer.reduce(result_id, all_variables)
        result = pool.diagram(result_id)
        total = sum(walker.revisit.values())
        from numpy import average
        avg = average(walker.revisit.values())
        print("Visits to {} nodes, total visits: {}, average: {}".format(len(walker.revisit), total, avg))
    # timer.start("Checking output")
    _check_output(diagram, result, variables)
    # timer.stop()
    return result.root_node.node_id


def _check_output(start_diagram, end_diagram, variables):
    present = set([str(v) for v in VariableFinder(start_diagram).walk()])
    remaining = set([str(v) for v in VariableFinder(end_diagram).walk()])
    variables = [str(v) for v in variables]
    if not remaining.issubset(present):
        raise RuntimeError("New variables ({}) have been introduced in the diagram (which contained {}) after summation"
                           .format(list(remaining - present), present))
    if not present.issubset(remaining | set(variables)):
        raise RuntimeError("Some variables ({}) have been erroneously removed: they are not present in the diagram "
                           "after summation ({}) nor in the variables to remove ({})"
                           .format(list(present - remaining - set(variables)), list(remaining), variables))
    if len(remaining & set(variables)) != 0:
        raise RuntimeError("Some variables ({}) that should be summed out ({}) still appear in the diagram after "
                           "summation. The diagram started out with variables {} and now contains {}"
                           .format(list(remaining & set(variables)), variables, list(present), list(remaining)),
                           end_diagram)
