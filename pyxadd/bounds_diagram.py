from pyxadd import build
from pyxadd import diagram
from pyxadd import view as exporting
from pyxadd import test
from pyxadd import operation
from pyxadd import leaf_transform
from pyxadd import reduce

import sympy


class BoundResolve(object):
    def __init__(self, pool, debug_path=None):
        self.pool = pool
        self.pool.add_var("_lb", "int")
        self.pool.add_var("_ub", "int")
        self.pool.add_var("_other_ubs", "int")
        self.pool.add_var("_other_lbs", "int")
        self.debug_path = debug_path
        self.ub_cache = None
        self.lb_cache = None
        self.builder = build.Builder(pool)

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

        integrated = leaf_transform.transform_leaves(symbolic_integrator, self.pool.diagram(node_id))
        self.export(self.pool.diagram(integrated), "integrated")
        result_id = self.resolve_lb_ub(integrated, var)
        self.ub_cache = None
        self.lb_cache = None
        return result_id

    def resolve_lb_ub(self, node_id, var, seen_ub=False, seen_lb=False):
        node = self.pool.get_node(node_id)
        b = self.builder
        # leaf
        if node.is_terminal():
            # TODO: to deal with unbounded constraints, we should either return 0 if we've seen bounds
            # or f(inf) if we haven't seen bounds
            return self.pool.zero_id
            # not leaf
        var_coefficient = node.test.operator.coefficient(var)
        if var_coefficient != 0:
            # Variable occurs in test

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

            # split into cases

            # case 1: root.test is smaller than all other UBs
            # -> we make root smaller than all other ubs, and we determine the lb
            ub_resolved_id = self.dag_resolve(var, operator, "leq", ub_branch, "ub", consume=False, notest=not seen_ub)
            best_ub = self.resolve_lb(ub_resolved_id, var, operator, seen_lb=seen_lb)
            self.export(self.pool.diagram(best_ub), "bestubU{}".format(node_id))

            # case 2: root.test is not going to be the ub, we need to make root || <- make root?
            # -> it needs to be larger than all other ubs; bound_max will append || <- lower than all?
            # a comparison of node.test.operator > ub(leaf) before each leaf.
            operator_bound = self.operator_to_bound(operator, var)
            bound_to_resolve = test.LinearTest("_other_ubs", ">=", operator_bound).operator
            recursive_bounds = self.resolve_lb_ub(ub_branch, var, seen_ub=True, seen_lb=seen_lb)
            some_ub = self.dag_resolve("_other_ubs", bound_to_resolve, "geq", recursive_bounds, "lb",
                                       notest=True, consume=not seen_ub)
            import time
            time_id = int(time.time() * 1000)
            self.export(self.pool.diagram(some_ub), "someubU{}_{}".format(node_id, time_id))

            # case 3: test is false, then root.test is the lower bound
            # -> make root.test larger than all other lbs
            lb_resolved_id = self.dag_resolve(var, (~operator).to_canonical(), "geq", lb_branch, "lb", consume=False,
                                              notest=not seen_lb)
            best_lb = self.resolve_ub(lb_resolved_id, var, (~operator).to_canonical(), seen_ub=seen_ub)
            self.export(self.pool.diagram(best_lb), "bestlbU{}".format(node_id))

            # case 4:
            # import pdb
            # pdb.set_trace()
            operator_bound = self.operator_to_bound((~operator).to_canonical(), var)
            bound_to_resolve = test.LinearTest("_other_lbs", "<=", operator_bound).operator
            recursive_bounds = self.resolve_lb_ub(lb_branch, var, seen_ub=seen_ub, seen_lb=True)
            # if node_id == 1899:
            #    import pdb
            #    pdb.set_trace()
            some_lb = self.dag_resolve("_other_lbs", bound_to_resolve, "leq", recursive_bounds, "ub",
                                       notest=True, consume=not seen_lb)
            self.export(self.pool.diagram(some_lb), "some_lbU{}".format(node_id))
            # if node_id == 1878:
            #    self.debug_path = "../../Dropbox/XADD Matrices/"
            #    testicol = self.resolve_lb_ub(node.child_false, var)
            #    exporting.export(self.pool.diagram(testicol), "../../Dropbox/XADD Matrices/testicol.dot", print_node_ids=True)
            # else:
            #    self.debug_path = None
            return (self.pool.diagram(some_lb)
                    + self.pool.diagram(best_lb)
                    + self.pool.diagram(some_ub)
                    + self.pool.diagram(best_ub)).root_id
        else:
            # Variable does not occur in test => recurse on both branches and merge
            test_node_id = self.pool.bool_test(node.test)

            true_branch_id = self.resolve_lb_ub(node.child_true, var, seen_ub=seen_ub, seen_lb=seen_lb)
            true_branch_diagram = self.pool.diagram(true_branch_id)

            false_branch_id = self.resolve_lb_ub(node.child_false, var, seen_ub=seen_ub, seen_lb=seen_lb)
            false_branch_diagram = self.pool.diagram(false_branch_id)

            result_id = b.ite(self.pool.diagram(test_node_id), true_branch_diagram, false_branch_diagram).root_id

            # DEBUG
            if node_id == 2283:
                print(node_id)
                result_node = self.pool.get_node(result_id)
                print(self.pool.get_node(node.child_true), self.pool.get_node(node.child_false))
                print(self.pool.get_node(result_node.child_true), self.pool.get_node(result_node.child_false))

            print("Resolve lb-ub (no resolve) result-id: {}".format(result_id))
            return result_id

    # view.export(pool.diagram(some_ub), "../../Dropbox/XADD Matrices/dr_1_someub_{}.dot".format(str(node.test.operator)))
    def resolve_ub(self, node_id, var, lower_bound, seen_ub=False):
        b = self.builder
        # if node_id == 70 and hash(str(lower_bound)) == 2594685873440222097:
        # import pdb
        # pdb.set_trace()
        node = self.pool.get_node(node_id)
        if node.is_terminal():
            return self.pool.zero_id
        print("Resolve_ub with {}x{} {}x{}".format(node.test.operator, node_id, lower_bound, hash(str(lower_bound))))
        self.export(self.pool.diagram(node_id), "ub_debugnid{}x{}".format(node_id, hash(str(lower_bound))))
        var_coefficient = node.test.operator.coefficient(var)
        if var_coefficient != 0:
            # Variable occurs in test

            if var_coefficient > 0:
                operator = node.test.operator
                ub_branch = node.child_true
                lb_branch = node.child_false
            else:
                operator = (~node.test.operator).to_canonical()
                ub_branch = node.child_false
                lb_branch = node.child_true

            best_ub = self.dag_resolve(var, operator, "leq", ub_branch, "ub", consume=True, notest=not seen_ub)

            bound_to_resolve = test.LinearTest("_other_ubs", ">=", self.operator_to_bound(operator, var)).operator
            recursive_bounds = self.resolve_ub(ub_branch, var, lower_bound, seen_ub=True)

            self.export(self.pool.diagram(recursive_bounds),
                        "resolve_ub_rec_bound_{}x{}".format(node_id, hash(str(lower_bound))))

            some_ub = self.dag_resolve("_other_ubs", bound_to_resolve, "geq", recursive_bounds, "lb",
                                       notest=True, consume=not seen_ub)

            self.export(self.pool.diagram(some_ub),
                        "resolve_ub_some_ub_{}x{}".format(node_id, hash(str(lower_bound))))

            non_ub = self.resolve_ub(lb_branch, var, lower_bound, seen_ub=seen_ub)

            resolve_test = self.resolve(var, lower_bound, "", operator, "ub")
            print("Resolve test (resolve_ub true branch) {}".format(self.pool.get_node(resolve_test)))
            res = self.builder.ite(self.pool.diagram(resolve_test),
                                   self.pool.diagram(best_ub) + self.pool.diagram(some_ub),
                                   self.pool.diagram(non_ub)
                                   )
            self.export(self.pool.diagram(res.root_id),
                        "ub_res_debugnid{}x{}".format(node_id, hash(str(lower_bound))))
            return res.root_id
        else:
            # Variable does not occur in test => recurse on both branches and merge
            test_node_id = self.pool.bool_test(node.test)

            true_branch_id = self.resolve_ub(node.child_true, var, lower_bound, seen_ub=seen_ub)
            true_branch_diagram = self.pool.diagram(true_branch_id)

            false_branch_id = self.resolve_ub(node.child_false, var, lower_bound, seen_ub=seen_ub)
            false_branch_diagram = self.pool.diagram(false_branch_id)

            result_id = b.ite(self.pool.diagram(test_node_id), true_branch_diagram, false_branch_diagram).root_id
            print("Resolve ub (no resolve) result-id: {}".format(result_id))
            return result_id

    def resolve_lb(self, node_id, var, upper_bound, seen_lb=False):
        b = self.builder
        node = self.pool.get_node(node_id)
        if node.is_terminal():
            return self.pool.zero_id
        print("Resolve_lb with {}x{} {}x{}".format(node.test.operator, node_id, upper_bound, hash(str(upper_bound))))
        self.export(self.pool.diagram(node_id), "reslb_debugnid{}x{}".format(node_id, hash(str(upper_bound))))
        var_coefficient = node.test.operator.coefficient(var)
        if var_coefficient != 0:
            if var_coefficient < 0:
                # True branch is upper-bound
                operator = node.test.operator
                lb_branch = node.child_true
                ub_branch = node.child_false
            else:
                # False branch is upper-bound
                operator = (~node.test.operator).to_canonical()
                lb_branch = node.child_false
                ub_branch = node.child_true

            # Compute first best lb and then some lb for the lb_branch
            best_lb = self.dag_resolve(var, operator, "geq", lb_branch, "lb", consume=True, notest=not seen_lb)
            operator_bound = self.operator_to_bound((~operator).to_canonical(), var)
            bound_to_resolve = test.LinearTest("_other_lbs", "<=", operator_bound).operator
            recursive_bounds = self.resolve_lb(lb_branch, var, upper_bound, seen_lb=True)
            some_lb = self.dag_resolve("_other_lbs", bound_to_resolve, "leq", recursive_bounds, "ub",
                                       notest=True, consume=not seen_lb)

            # For the ub_branch skip the test and recurse on the child
            non_lb = self.resolve_lb(ub_branch, var, upper_bound, seen_lb=seen_lb)
            resolve_test_id = self.resolve(var, upper_bound, "", operator, "lb")
            print("Resolve test (resolve_lb true branch) {}".format(self.pool.get_node(resolve_test_id)))
            # || <- What is about to happen here?
            res = self.builder.ite(self.pool.diagram(resolve_test_id),
                                   self.pool.diagram(best_lb) + self.pool.diagram(some_lb),
                                   self.pool.diagram(non_lb)
                                   )
            return res.root_id
        else:
            # Variable does not occur in test => recurse on both branches and merge
            test_node_id = self.pool.bool_test(node.test)
            true_branch_id = self.resolve_lb(node.child_true, var, upper_bound, seen_lb=seen_lb)
            true_branch_diagram = self.pool.diagram(true_branch_id)

            false_branch_id = self.resolve_lb(node.child_false, var, upper_bound, seen_lb)
            false_branch_diagram = self.pool.diagram(false_branch_id)

            result_id = b.ite(self.pool.diagram(test_node_id), true_branch_diagram, false_branch_diagram).root_id
            print("Resolve lb (no resolve) result-id: {}".format(node_id))
            return result_id

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
        :param pyxadd.test.LinearTest linear_test: The test
        :param pyxadd.diagram.Diagram true_diagram: The diagram for the true case
        :param pyxadd.diagram.Diagramfalse_diagram: The diagram for the false case
        :return pyxadd.diagram.Diagram: The if-then-else diagram
        """
        if linear_test.operator.is_tautology():
            if linear_test.operator.rhs < 0:
                # Infeasible
                return false_diagram
            else:
                # Tautology
                return true_diagram
        else:
            return self.builder.ite(self.pool.diagram(self.pool.bool_test(linear_test)), true_diagram, false_diagram)

    def dag_resolve(self, var, operator, direction, node_id, bound_type, notest=False, consume=False):
        node = self.pool.get_node(node_id)

        if node.is_terminal():

            print operator, var
            bounded_exp = node.expression.subs({"_" + bound_type: self.operator_to_bound(operator, var)})
            bound = self.operator_to_bound(operator, var)
            bound_test = (test.LinearTest("_other_ubs", ">", bound) if bound_type == "ub"
                          else test.LinearTest("_other_lbs", "<", bound))
            if notest:
                res = self.pool.terminal(
                    node.expression.subs({"_" + bound_type: self.operator_to_bound(operator, var)}))
            else:
                res = self.simplify(bound_test,
                                    self.builder.exp(bounded_exp),
                                    self.builder.exp(sympy.sympify("0"))).root_id
            return res
        var_coefficient = node.test.operator.coefficient(var)
        if var_coefficient != 0:
            test_node_id = self.pool.bool_test(node.test)
            resolve_true = self.resolve(var, operator, direction, node.test.operator, bound_type)
            resolve_false = self.resolve(var, operator, direction, (~node.test.operator).to_canonical(), bound_type)
            dr_true = self.dag_resolve(var, operator, direction, node.child_true,
                                       bound_type, notest=notest, consume=consume)
            dr_false = self.dag_resolve(var, operator, direction, node.child_false,
                                        bound_type, notest=notest, consume=consume)
            if consume:
                if resolve_true is not None:
                    res = self.builder.ite(self.pool.diagram(resolve_true),
                                           self.pool.diagram(dr_true),
                                           self.pool.diagram(dr_false)
                                           )
                else:
                    # is terminal
                    # not matching => terminal(1)
                    res = self.builder.ite(self.pool.diagram(resolve_false),
                                           self.pool.diagram(dr_false),
                                           self.pool.diagram(dr_true)
                                           )
            else:
                one_diagram = self.builder.exp(1)
                resolve_true_diagram = one_diagram if resolve_true is None else self.pool.diagram(resolve_true)
                resolve_false_diagram = one_diagram if resolve_false is None else self.pool.diagram(resolve_false)
                res = self.builder.ite(self.pool.diagram(test_node_id),
                                       resolve_true_diagram * self.pool.diagram(dr_true),
                                       resolve_false_diagram * self.pool.diagram(dr_false)
                                       )

            print("Dag resolve {} {} {} {}".format(operator, hash(str(operator)), direction, node))
            print(self.pool.get_node(node.child_true), self.pool.get_node(node.child_false))
            print("resolved {}".format(self.pool.get_node(res.root_id)))
            if not res.root_node.is_terminal():
                print(self.pool.get_node(self.pool.get_node(res.root_id).child_true),
                      self.pool.get_node(self.pool.get_node(res.root_id).child_false))
            self.export(res, "dr_debugnid{}x{}".format(node_id, hash(str(operator))))
            return res.root_id
        else:
            test_node_id = self.pool.bool_test(node.test)
            result_id = self.pool.apply(operation.Summation, self.pool.apply(operation.Multiplication, test_node_id,
                                                                             self.dag_resolve(var, operator, direction,
                                                                                              node.child_true,
                                                                                              bound_type,
                                                                                              consume=consume,
                                                                                              notest=notest)),
                                        self.pool.apply(operation.Multiplication, self.pool.invert(test_node_id),
                                                        self.dag_resolve(var, operator, direction, node.child_false,
                                                                         bound_type, consume=consume,
                                                                         notest=notest)))
            print("Dag resolve (no resolve) result-id: {}".format(node_id))
            return result_id

    def resolve(self, var, operator_lhs, direction, operator_rhs, bound_type):
        # print operator_rhs
        # pdb.set_trace()
        operator_rhs = operator_rhs.to_canonical()
        operator_lhs = operator_lhs.to_canonical()
        rhs_coefficient = operator_rhs.coefficient(var)
        print("Resolving ({}) {} ({}) ({})".format(repr(operator_lhs), direction, repr(operator_rhs), bound_type))
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
            print("Resolving ({}) {} ({}) = ({})".format(repr(operator_lhs), direction, repr(operator_rhs), repr(res)))
            if res.is_tautology():
                if res.rhs < 0:
                    return self.pool.terminal(0)
                else:
                    return self.pool.terminal(1)
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
