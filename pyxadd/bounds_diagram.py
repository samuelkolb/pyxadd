from pyxadd import build
from pyxadd import diagram
from pyxadd import view as exporting
from pyxadd import test
from pyxadd import operation
from pyxadd import leaf_transform
from pyxadd import reduce

import sympy


class BoundResolve(object):
    def __init__(self, pool, debug_path=None, cache_result=True):
        self.pool = pool
        self.pool.add_var("_lb", "int")
        self.pool.add_var("_ub", "int")
        self.pool.add_var("_other_ubs", "int")
        self.pool.add_var("_other_lbs", "int")
        self.debug_path = debug_path
        # self.debug_path = "../../Dropbox/XADD Matrices/"
        self.cache_hits = 0
        self.cache_calls = 0
        self.ub_cache = None
        self.lb_cache = None
        self.builder = build.Builder(pool)
        self.cache_result = cache_result
        self.resolve_cache = None

    def export(self, diagram_to_export, name):
        if self.debug_path is not None:
            if not name.endswith(".dot"):
                name += ".dot"
            exporting.export(diagram_to_export, "{}/{}".format(self.debug_path, name), print_node_ids=True)

    def integrate(self, node_id, var):
        self.ub_cache = {}
        self.lb_cache = {}

        def symbolic_integrator(terminal_node, d):
            assert isinstance(terminal_node, diagram.TerminalNode)
            return d.pool.terminal(sympy.Sum(terminal_node.expression, (var, "_lb", "_ub")).doit())

        if self.pool.get_var_type(var) != "bool":
            integrated = leaf_transform.transform_leaves(symbolic_integrator, self.pool.diagram(node_id))
        else:
            integrated = node_id
        self.export(self.pool.diagram(integrated), "integrated")
        self.resolve_cache = dict()
        result_id = self.resolve_lb_ub(integrated, var)
        # result_id = order.order(self.pool.diagram(self.resolve_lb_ub(integrated, var))).root_id
        self.resolve_cache = None

        self.ub_cache = None
        self.lb_cache = None

        return result_id

    def resolve_lb_ub(self, node_id, var, ub=None, lb=None, rl=0):
        method = "no_reduce"  # "fast_smt"
        prefix = rl * "." + "({})({})({})".format(node_id, ub, lb)
        # print(prefix + " enter")
        if self.cache_result:
            key = (node_id, ub, lb)
            self.cache_calls += 1
            # print key, self.cache_calls, self.cache_hits
            if key in self.resolve_cache:
                self.cache_hits += 1
                # print("Cache hit for key={}".format(key))
                return self.resolve_cache[key]

        def cache_result(result):
            if self.cache_result:
                self.resolve_cache[key] = result
            return result

        node = self.pool.get_node(node_id)
        b = self.builder
        # print "ub_lb_resolve node: {}, ub: {}, lb: {}, {} : {}".format(node, ub, lb, hash(str(ub)), hash(str(lb)))
        # leaf
        if node.is_terminal():
            if self.pool.get_var_type(var) == "bool":
                return self.pool.terminal(2 * node.expression)
            if ub is None or lb is None:
                # TODO: to deal with unbounded constraints, we should either return 0 if we've seen bounds
                # or f(inf) if we haven't seen bounds
                return cache_result(self.pool.zero_id)
            else:
                ub_sub = self.operator_to_bound(ub, var)
                try:  # TODO Check rounding
                    ub_sub = int(ub_sub)
                except TypeError:
                    print("Could not convert ub {} to int".format(ub_sub))
                    raise

                lb_sub = self.operator_to_bound(lb, var)
                try:
                    lb_sub = int(lb_sub)  # TODO ROUND UP NOT DOWN
                except TypeError:
                    print("Could not convert lb {} to int".format(lb_sub))
                    raise
                # assert len(ub_sub.free_symbols) > 0 or ub_sub == int(ub_sub), "ub is float: {}".format(ub_sub)
                # assert len(lb_sub.free_symbols) > 0 or lb_sub == int(lb_sub), "lb is float: {}".format(lb_sub)
                bounded_exp = node.expression.subs({"_ub": ub_sub, "_lb": lb_sub})
                res = self.pool.terminal(bounded_exp)
                # print "->", self.pool.get_node(res)
                return cache_result(res)
                # not leaf

        if var in node.test.variables:
            # Variable occurs in test

            if self.pool.get_var_type(var) == "bool":
                from pyxadd.operation import Summation
                return self.pool.apply(Summation, node.child_true, node.child_false)

            var_coefficient = node.test.operator.coefficient(var)
            if var_coefficient > 0:
                # True branch is upper-bound
                operator = node.test.operator
                ub_branch = node.child_true
                lb_branch = node.child_false
            else:
                # False branch is upper-bound
                operator = (~node.test.operator).to_canonical()
                ub_branch = node.child_false
                lb_branch = node.child_true
            # ub_at_node = self.operator_to_bound(operator, var)
            # lb_at_node = self.operator_to_bound((~operator).to_canonical(), var)
            ub_at_node = operator
            lb_at_node = (~operator).to_canonical()
            pass_ub = False
            if lb is not None:
                ub_expr = self.operator_to_bound(ub_at_node, var)
                lb_expr = self.operator_to_bound(lb, var)
                ub_comp = (ub_expr >= lb_expr)
                if ub_comp is sympy.S.false:
                    # this branch is infeasible
                    ub_consistency = self.pool.zero_id
                    some_or_best_ub = self.pool.diagram(self.pool.zero_id)
                    pass_ub = True
                elif ub_comp is sympy.S.true:
                    ub_consistency = self.pool.one_id
                else:
                    ub_consistency = self.pool.bool_test(test.LinearTest(ub_expr, ">=", lb_expr))
            else:
                ub_consistency = self.pool.one_id
            if ub is not None and not pass_ub:
                ub_test = self.resolve(var, ub, "geq", ub_at_node, "ub")
                if ub_test == self.pool.one_id:
                    some_ub = self.pool.zero_id
                else:
                    some_ub = self.resolve_lb_ub(ub_branch, var, ub=ub, lb=lb, rl=rl + 1)
                if ub_test == self.pool.zero_id:
                    best_ub = self.pool.zero_id
                else:
                    best_ub = self.resolve_lb_ub(ub_branch, var, ub=ub_at_node, lb=lb, rl=rl + 1)
                best_ub = self.pool.diagram(best_ub).reduce(method=method).root_id  # RED
                some_ub = self.pool.diagram(some_ub).reduce(method=method).root_id  # RED
                some_or_best_ub = self.simplify(ub_test,
                                                self.pool.diagram(best_ub),
                                                self.pool.diagram(some_ub))
                # ub_test = self.pool.bool_test(test.LinearTest(ub, ">", ub_at_node))
            elif not pass_ub:
                best_ub = self.resolve_lb_ub(ub_branch, var, ub=ub_at_node, lb=lb, rl=rl + 1)
                some_or_best_ub = self.pool.diagram(best_ub)
            # print(prefix + " ub done")
            pass_lb = False
            if ub is not None:
                ub_expr = self.operator_to_bound(ub, var)
                lb_expr = self.operator_to_bound(lb_at_node, var)
                lb_comp = (ub_expr >= lb_expr)
                if lb_comp is sympy.S.false:
                    # this branch is infeasible
                    lb_consistency = self.pool.zero_id
                    some_or_best_lb = self.pool.diagram(self.pool.zero_id)
                    pass_lb = True
                elif lb_comp is sympy.S.true:
                    lb_consistency = self.pool.one_id
                else:
                    lb_consistency = self.pool.bool_test(test.LinearTest(ub_expr, ">=", lb_expr))
            else:
                lb_consistency = self.pool.one_id
            if lb is not None and not pass_lb:
                lb_test = self.resolve(var, lb_at_node, "geq", lb, "lb")
                if lb_test == self.pool.one_id:
                    some_lb = self.pool.zero_id
                else:
                    some_lb = self.resolve_lb_ub(lb_branch, var, ub=ub, lb=lb, rl=rl + 1)
                if lb_test == self.pool.zero_id:
                    best_lb = self.pool.zero_id
                else:
                    best_lb = self.resolve_lb_ub(lb_branch, var, ub=ub, lb=lb_at_node, rl=rl + 1)
                best_lb = self.pool.diagram(best_lb).reduce(method=method).root_id  # RED
                some_lb = self.pool.diagram(some_lb).reduce(method=method).root_id  # RED
                some_or_best_lb = self.simplify(lb_test,
                                                self.pool.diagram(best_lb),
                                                self.pool.diagram(some_lb))
            elif not pass_lb:
                best_lb = self.resolve_lb_ub(lb_branch, var, ub=ub, lb=lb_at_node, rl=rl + 1)
                some_or_best_lb = self.pool.diagram(best_lb)

            lb_branch = some_or_best_lb * self.pool.diagram(lb_consistency)
            ub_branch = some_or_best_ub * self.pool.diagram(ub_consistency)

            # print(prefix + " lb done")
            lb_branch = lb_branch.reduce(method=method)  # RED
            ub_branch = ub_branch.reduce(method=method)  # RED
            res = (lb_branch + ub_branch)
            # self.export(res, "res{}_{}_{}".format(node_id, hash(str(ub)), hash(str(lb))))
            return cache_result(res.root_id)
        else:
            test_node_id = self.pool.bool_test(node.test)
            true_branch_id = self.resolve_lb_ub(node.child_true, var, ub=ub, lb=lb)
            true_branch_diagram = self.pool.diagram(true_branch_id)
            false_branch_id = self.resolve_lb_ub(node.child_false, var, ub=ub, lb=lb)
            false_branch_diagram = self.pool.diagram(false_branch_id)

            result_id = b.ite(self.pool.diagram(test_node_id), true_branch_diagram, false_branch_diagram).root_id
            return cache_result(result_id)

    # view.export(pool.diagram(some_ub), "../../Dropbox/XADD Matrices/dr_1_someub_{}.dot".format(str(node.test.operator)))

    def to_exp(self, op, var):
        expression = sympy.sympify(op.rhs)
        for k, v in op.lhs.items():
            if k != var:
                expression = -sympy.S(k) * v + expression
        return expression

    def operator_to_bound(self, operator, var):
        """
        Converts the given operator to a bound on the given variable
        :param pyxadd.test.Operator operator: The operator to convert
        :param str var:
        :return: Bound expression (sympy)
        """
        # TODO Check for integer division
        if var not in operator.lhs:
            return sympy.S(1)
        bound = operator.times(1 / operator.coefficient(var)).weak()
        exp_pos = self.to_exp(bound, var)
        return exp_pos

    def simplify(self, linear_test, true_diagram, false_diagram):
        """
        Simplifies an if-then-else in case the test is a tautology
        :param int linear_test: The test
        :param pyxadd.diagram.Diagram true_diagram: The diagram for the true case
        :param pyxadd.diagram.Diagramfalse_diagram: The diagram for the false case
        :return pyxadd.diagram.Diagram: The if-then-else diagram
        """
        if linear_test == self.pool.one_id:
            return true_diagram

        if linear_test == self.pool.zero_id:
            return false_diagram

        operator = self.pool.get_node(linear_test).test.operator.to_canonical()
        if operator.is_tautology():
            if operator.rhs < 0:
                # Infeasible
                return false_diagram
            else:
                # Tautology
                return true_diagram
        else:
            return self.builder.ite(self.pool.diagram(linear_test), true_diagram, false_diagram)

    def resolve(self, var, operator_lhs, direction, operator_rhs, bound_type):
        # print operator_rhs
        # pdb.set_trace()
        operator_rhs = operator_rhs.to_canonical()
        operator_lhs = operator_lhs.to_canonical()
        rhs_coefficient = operator_rhs.coefficient(var)
        # print("Resolving ({}) {} ({}) ({})".format(repr(operator_lhs), direction, repr(operator_rhs), bound_type))
        rhs_type = "na"
        if rhs_coefficient > 0:
            rhs_type = "ub"
        elif rhs_coefficient < 0:
            rhs_type = "lb"
        else:
            raise RuntimeError("Variable {} does not appear in expression {}".format(var, operator_lhs))
        if rhs_type != bound_type:
            return None
            # return self.pool.terminal(1)
        else:
            if bound_type == "ub":
                if direction == "geq":
                    res = operator_lhs.resolve(var, operator_rhs.switch_direction())
                elif direction == "leq":
                    res = operator_rhs.resolve(var, operator_lhs.switch_direction())
                else:
                    res = operator_lhs.resolve(var, operator_rhs)
            elif bound_type == "lb":
                if direction == "geq":
                    res = operator_rhs.resolve(var, operator_lhs.switch_direction())
                elif direction == "leq":
                    res = operator_lhs.resolve(var, operator_rhs.switch_direction())
                else:
                    res = operator_lhs.resolve(var, operator_rhs)
            res = res.to_canonical()
            # print ",".join([str(u) for u in [operator_rhs, operator_lhs, res]])
            # print("Resolving ({}) {} ({}) = ({})".format(repr(operator_lhs), direction, repr(operator_rhs), repr(res)))
            if res.is_tautology():
                if res.rhs < 0:
                    return self.pool.zero_id
                else:
                    return self.pool.one_id
            # zero = True
            # for var in res.variables:
            #   if res.coefficient != 0:
            #     zero = False
            #     break
            # if zero:
            #   return pool.terminal(1)
            return self.pool.bool_test(test.LinearTest(res))


if __name__ == "__main__":
    the_pool = diagram.Pool()


    def two_var_diagram():
        import pdb
        # pdb.set_trace()
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 1)
        bounds &= b.test("y", ">=", 1) & b.test("y", "<=", 3)
        two = b.test("x", ">=", "y")
        return b.ite(bounds, b.ite(two, b.terminal("x"), b.terminal("10")), b.terminal(0))


    b = build.Builder(the_pool)
    b.ints("x", "y", "a", "b", "c", "_ub", "_lb", "bla")
    diagram1 = b.ite(b.test("x", "<=", "a"),
                     b.ite(b.test("x", ">=", "b"),
                           b.exp("_ub - _lb"), b.exp(0)),
                     b.ite(b.test("x", "<=", "c"),
                           b.exp("(_ub - _lb)**2"), b.exp(0))
                     )
    diagram2 = b.ite(b.test("x", ">=", "b"), b.exp("_ub - _lb"), b.exp(0))
    bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
    # d = b.ite(bounds, b.terminal("x"), b.terminal(0))

    d = two_var_diagram()

    operator_1 = test.LinearTest("x", "<=", "a").operator
    operator_2 = test.LinearTest("x", "<=", "2").operator
    operator_3 = test.LinearTest("x", ">=", "a").operator
    operator_4 = test.LinearTest("x", ">=", "2").operator

    bound_resolve = BoundResolve(the_pool)
    resolved_node_id = bound_resolve.resolve("x", operator_1, "leq", operator_2, "ub")
    print(the_pool.get_node(resolved_node_id).test.operator)

    resolved_node_id = bound_resolve.resolve("x", operator_1, "geq", operator_2, "ub")

    print(the_pool.get_node(resolved_node_id).test.operator)

    resolved_node_id = bound_resolve.resolve("x", operator_3, "leq", operator_4, "lb")
    print(the_pool.get_node(resolved_node_id).test.operator)

    resolved_node_id = bound_resolve.resolve("x", operator_3, "geq", operator_4, "lb")

    print(the_pool.get_node(resolved_node_id).test.operator)
    # dr = dag_resolve("x", operator_1, pool.bool_test(test.LinearTest(operator_2)), "geq", "ub")
    # print("Diagram is {}ordered".format("" if order.is_ordered(pool.diagram(dr)) else "not "))
    # view.export(pool.diagram(dr), "../../Dropbox/XADD Matrices/test.dot")
    test_diagram = d
    bound_resolve.export(test_diagram, "diagram")
    # dr = dag_resolve("x", operator_1, diagram.root_id, "leq", "ub")
    # view.export(pool.diagram(dr), "../../Dropbox/XADD Matrices/dr.dot")
    # recurse(diagram.root_id)
    dr = bound_resolve.integrate(test_diagram.root_id, "x")
    bound_resolve.export(the_pool.diagram(dr), "result")

    dr = reduce.LinearReduction(the_pool).reduce(dr)
    # fm = fourier_motzkin(bounds.root_id, "ub")
    bound_resolve.export(the_pool.diagram(dr), "result_reduced")

    d_const = the_pool.diagram(dr)
    for y in range(-20, 20):
        s = 0
        for x in range(-20, 20):
            s += d.evaluate({"x": x, "y": y})
        print(y, ":", s - d_const.evaluate({"y": y}))
