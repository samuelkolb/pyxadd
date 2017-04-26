from __future__ import print_function

import os
import unittest

import math

from pyxadd import matrix_vector, bounds_diagram
from pyxadd.build import Builder
from pyxadd.diagram import Diagram, Pool
from pyxadd.matrix_vector import SummationWalker, matrix_multiply
from pyxadd.partial import PartialWalker
from pyxadd.reduce import LinearReduction
from pyxadd.test import LinearTest
from pyxadd import timer
from tests import export


class TestMatrixVector(unittest.TestCase):
    resolve_export = export.Exporter(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"),
        "resolve",
        True)

    def setUp(self):
        self.diagram = TestMatrixVector.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool()
        pool.int_var("x", "y")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 8) & b.test("y", ">=", 1) & b.test("y", "<=", 10)
        return bounds * b.ite(b.test("x", ">=", "y"), b.terminal("2*x + 3*y"), b.terminal("3*x + 2*y"))

    # FIXME In case I forget: Introduce ordering on integer comparisons

    def test_summation_one_var(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
        d = b.ite(bounds, b.terminal("x"), b.terminal(0))
        d_const = Diagram(pool, SummationWalker(d, "x").walk())
        # from pyxadd import bounds_diagram
        # d_const = pool.diagram(bounds_diagram.BoundResolve(pool).integrate(d.root_id, "x"))
        self.assertEqual(55, d_const.evaluate({}))

    def test_summation_two_var(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
        bounds &= b.test("y", ">=", 0) & b.test("y", "<=", 1)
        d = b.ite(bounds, b.terminal("x"), b.terminal(0))
        d_const = Diagram(pool, SummationWalker(d, "x").walk())
        # from pyxadd import bounds_diagram
        # d_const = pool.diagram(bounds_diagram.BoundResolve(pool).integrate(d.root_id, "x"))
        for y in range(2):
            self.assertEqual(55, d_const.evaluate({"y": y}))

    def test_summation_two_var_test(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 1)
        bounds &= b.test("y", ">=", 1) & b.test("y", "<=", 3)
        two = b.test("x", ">=", "y")
        d = b.ite(bounds, b.ite(two, b.terminal("x"), b.terminal("10")), b.terminal(0))

        summed = Diagram(pool, SummationWalker(d, "x").walk())
        from pyxadd import bounds_diagram
        summed = pool.diagram(bounds_diagram.BoundResolve(pool).integrate(d.root_id, "x"))
        d_const = summed.reduce(["y"])
        for y in range(-20, 20):
            s = 0
            for x in range(-20, 20):
                s += d.evaluate({"x": x, "y": y})
            self.assertEqual(s, d_const.evaluate({"y": y}), msg="Expected ({}) and obtained ({}) differ at y = {}"
                             .format(s, d_const.evaluate({"y": y}), y))

    def test_mixed_symbolic(self):
        pool = self.diagram.pool
        diagram_y = Diagram(pool, SummationWalker(self.diagram, "x").walk())
        # from pyxadd import bounds_diagram
        # diagram_y = pool.diagram(bounds_diagram.BoundResolve(pool).integrate(diagram_y.root_id, "x"))

        diagram_y = Diagram(diagram_y.pool, LinearReduction(diagram_y.pool).reduce(diagram_y.root_node.node_id, ["y"]))

        for y in range(0, 12):
            row_result = 0
            for x in range(0, 12):
                row_result += self.diagram.evaluate({"x": x, "y": y})
            self.assertEqual(diagram_y.evaluate({"y": y}), row_result)

    def test_partial(self):
        partial = PartialWalker(self.diagram, {"y": 2}).walk()
        for x in range(-10, 10):
            if x < 0 or x > 8:
                self.assertEqual(0, partial.evaluate({"x": x}))
            elif x > 2:
                self.assertEqual(2 * x + 6, partial.evaluate({"x": x}))
            else:
                self.assertEqual(3 * x + 4, partial.evaluate({"x": x}))

    def _test_bounds_resolve_1(self):
        import os
        from tests import test_evaluate
        from pyxadd import bounds_diagram
        from pyxadd import variables
        from tests import export

        exporter = export.Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "resolve", True)
        diagram_1, vars_1 = test_evaluate.get_diagram_1()
        exporter.export(diagram_1, "diagram")
        pool = diagram_1.pool
        b = Builder(pool)
        resolve = bounds_diagram.BoundResolve(pool)
        result_id = diagram_1.root_id
        control_id = diagram_1.root_id
        reducer = LinearReduction(pool)
        c = 1  # 000.0
        for var in vars_1:
            var_name = str(var[0])
            if var_name == "c_f1":
                self.compare_results_paths(pool.diagram(result_id), var_name)
            with open("root_{}.txt".format(var_name), "w") as stream:
                print(str(result_id), file=stream)
            with open("diagram_{}.txt".format(var_name), "w") as stream:
                print(Pool.to_json(b.pool), file=stream)
            result_id = resolve.integrate(result_id, var_name)
            control_id = matrix_vector.sum_out(pool, control_id, [var_name])
            result_diagram = pool.diagram(result_id)
            control_diagram = pool.diagram(control_id)
            # result_diagram = pool.diagram(reducer.reduce(result_diagram.root_id))
            # control_diagram = pool.diagram(reducer.reduce(control_diagram.root_id))
            difference_diagram = pool.diagram(reducer.reduce((result_diagram - control_diagram).root_id))
            exporter.export(result_diagram, "resolve_without_{}".format(var_name))
            exporter.export(control_diagram, "control_without_{}".format(var_name))
            exporter.export(difference_diagram, "difference_without_{}".format(var_name))
            result_id = (b.terminal(1 / c) * result_diagram).root_id
            control_id = (b.terminal(1 / c) * control_diagram).root_id
            self.assertTrue(var_name not in variables.variables(result_diagram), "{} not eliminated".format(var_name))
            self.assertTrue(var_name not in variables.variables(control_diagram), "{} not eliminated".format(var_name))

        self.assertTrue(len(variables.variables(result_diagram)) == 0)
        self.assertTrue(len(variables.variables(control_diagram)) == 0)
        print(control_diagram.evaluate({}), result_diagram.evaluate({}))
        self.assertEquals(control_diagram.evaluate({}), result_diagram.evaluate({}))

    def _test_bounds_resolve_final_variable(self):
        import os
        from tests import test_evaluate
        from pyxadd import bounds_diagram
        from pyxadd import variables
        from tests import export

        exporter = export.Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "resolve", True)

        # Import pool file
        pool_file = "data/bound_resolve/diagram_r_f0.txt"
        root_id = 1663

        with open(pool_file, "r") as stream:
            json_input = stream.readline()

        exported_pool = Pool.from_json(json_input)
        diagram_1 = exported_pool.diagram(root_id)
        vars_1 = [('c_f1', 0, 964)]

        exporter.export(diagram_1, "diagram")
        pool = diagram_1.pool
        b = Builder(pool)
        resolve = bounds_diagram.BoundResolve(pool)
        result_id = diagram_1.root_id
        control_id = diagram_1.root_id
        reducer = LinearReduction(pool)
        c = 1000.0
        for var in vars_1:
            var_name = str(var[0])
            result_id = resolve.integrate(result_id, var_name)
            control_id = matrix_vector.sum_out(pool, control_id, [var_name])
            result_diagram = pool.diagram(result_id)
            control_diagram = pool.diagram(control_id)
            result_diagram = pool.diagram(reducer.reduce(result_diagram.root_id))
            control_diagram = pool.diagram(reducer.reduce(control_diagram.root_id))
            difference_diagram = pool.diagram(reducer.reduce((result_diagram - control_diagram).root_id))
            exporter.export(result_diagram, "resolve_without_{}".format(var_name))
            exporter.export(control_diagram, "control_without_{}".format(var_name))
            exporter.export(difference_diagram, "difference_without_{}".format(var_name))
            result_id = (b.terminal(1 / c) * result_diagram).root_id
            control_id = (b.terminal(1 / c) * control_diagram).root_id
            self.assertTrue(var_name not in variables.variables(result_diagram), "{} not eliminated".format(var_name))
            self.assertTrue(var_name not in variables.variables(control_diagram), "{} not eliminated".format(var_name))
        self.assertTrue(len(variables.variables(result_diagram)) == 0)
        self.assertTrue(len(variables.variables(control_diagram)) == 0)
        self.assertEquals(control_diagram.evaluate({}), result_diagram.evaluate({}))

    def test_bounds_resolve_some_ub_1(self):
        import os
        from tests import export
        from pyxadd import bounds_diagram

        exporter = export.Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "resolve", True)
        b = Builder()
        b.ints("x", "b", "c", "d", "y")
        zero_test = b.test("x", ">=", 0)
        b_test = b.test("x", "<=", "b")
        y_test = b.test("y", "<=", 10)
        c_test = b.test("x", "<=", "c")
        d_test = b.test("x", "<=", "d")
        d = zero_test * b_test * b.ite(y_test,
                                       c_test * b.exp("3"),
                                       d_test * b.exp("11"))
        exporter.export(d, "some_ub_diagram")
        resolve = bounds_diagram.BoundResolve(b.pool, "./visual/resolve/debug/")
        result_id = resolve.integrate(d.root_id, "x")
        result_diagram = b.pool.diagram(result_id)
        exporter.export(result_diagram, "some_ub_result")
        reducer = LinearReduction(b.pool)
        reduced_result = b.pool.diagram(reducer.reduce(result_id))
        exporter.export(reduced_result, "some_ub_result_reduced")

        control_id = matrix_vector.sum_out(b.pool, d.root_id, ["x"])
        control_diagram = b.pool.diagram(control_id)
        exporter.export(control_diagram, "some_ub_control")

        reduced_control = b.pool.diagram(reducer.reduce(control_id))
        exporter.export(reduced_control, "some_ub_control_reduced")

    def test_bounds_resolve_numerical_bounds(self):
        b = Builder()
        b.ints("x")
        test_10 = b.test("x", "<=", 10)
        test_5 = b.test("x", "<=", 5)
        test_6 = b.test("x", ">=", 6)
        test_13 = b.test("x", "<=", 13)
        test_15 = b.test("x", ">=", 15)
        test_0 = b.test("x", ">=", 0)
        test_20 = b.test("x", "<=", 20)

        # Dis-joined paths
        branch_upper_bounds = test_10 * test_5 * test_13 * test_0 * b.exp(11)
        branch_lower_bounds = ~test_10 * test_6 * test_15 * test_20 * b.exp(77)

        test_diagram = branch_upper_bounds + branch_lower_bounds
        correct_result = 11 * (5 - 0 + 1) + 77 * (20 - 15 + 1)

        self.compare_results(test_diagram, "x", correct_result)

        # Rejoining path
        shared = b.ite(b.test("x", "<=", 16), b.exp(11), b.exp(77))

        branch_upper_bounds = test_10 * test_5 * test_13 * test_0 * shared
        branch_lower_bounds = ~test_10 * test_6 * test_15 * test_20 * shared

        test_diagram = branch_upper_bounds + branch_lower_bounds
        correct_result = 11 * (5 - 0 + 1) + 77 * (20 - 17 + 1) + 11 * (16 - 15 + 1)

        self.compare_results(test_diagram, "x", correct_result)

        paths = self.get_paths(test_diagram)
        for i in range(len(paths)):
            path = paths[i]
            self.resolve_export.export(path, "path{}".format(i))

    def test_bounds_resolve_inconsistent_branches(self):
        b = Builder()
        b.ints("x")
        test_10 = b.test("x", "<=", 10)
        test_20 = b.test("x", "<=", 20)
        # test_0 = b.test("x", "<=", 0)
        test_5 = b.test("x", "<=", 5)

        # test_diagram_infinite = b.ite(test_10, b.ite(test_0, 0, b.ite(test_5, 5, 7)), 11)
        # self.compare_results(test_diagram_infinite)

        test_diagram = b.ite(test_10, b.ite(test_20, 0, b.ite(test_5, 5, 7)), test_20 * b.exp(11))
        self.compare_results(test_diagram, "x", (20 - 11 + 1) * 11)

    def test_bounds_resolve_sub_diagram(self):
        b = Builder()
        b.ints("c_f1")
        test_2 = b.test("c_f1", "<=", 2)
        test_5 = b.test("c_f1", "<=", 5)
        test_10 = b.test("c_f1", "<=", 10)
        test_16 = b.test("c_f1", "<=", 16)
        test_9 = b.test("c_f1", "<=", 9)
        test_21 = b.test("c_f1", "<=", 21)
        test_28 = b.test("c_f1", "<=", 28)
        test_964 = b.test("c_f1", "<=", 964)

        val_1 = 2616157026.45477
        leaf_1 = b.exp(val_1)
        val_2 = 2615431790.14078
        leaf_2 = b.exp(val_2)

        path_1 = ~test_2 * ~test_5 * ~test_10 * ~test_16 * ~test_9 * test_21 * test_28 * test_964 * leaf_1
        path_2 = ~test_2 * ~test_5 * ~test_10 * test_16 * ~test_9 * test_21 * test_28 * test_964 * leaf_2

        test_diagram = path_1 + path_2
        result = (21 - 17 + 1) * val_1 + (16 - 11 + 1) * val_2
        self.compare_results(test_diagram, "c_f1", result)

    def test_bounds_resolve_sub_diagram_simplified(self):
        b = Builder()
        b.ints("c_f1")
        test_2 = b.test("c_f1", "<=", 2)
        test_10 = b.test("c_f1", "<=", 10)
        test_16 = b.test("c_f1", "<=", 16)
        test_9 = b.test("c_f1", "<=", 9)
        test_21 = b.test("c_f1", "<=", 21)

        val_1 = 2616157026.45477
        leaf_1 = b.exp(val_1)
        val_2 = 2615431790.14078
        leaf_2 = b.exp(val_2)

        # First false test => unbinding lower-bound
        path_1 = ~test_2 * ~test_10 * ~test_16 * ~test_9 * test_21 * leaf_1
        path_2 = ~test_2 * ~test_10 * test_16 * ~test_9 * test_21 * leaf_2

        test_diagram = path_1 + path_2
        result = (21 - 17 + 1) * val_1 + (16 - 11 + 1) * val_2
        self.compare_results(test_diagram, "c_f1", result)

    def test_xor(self):
        def build_xor(n):
            import random
            test_dict = {}
            b = Builder()
            b.ints("x", "c")

            bounds = b.test("x", "<=", 100) * b.test("x", ">=", 0) * b.test("c", "<=", 100) * b.test("c", ">=", 0)
	    #for i in range(n):
            #    constant = "c{}".format(i + 1)
            #    b.ints(constant)
            #    bounds = bounds * b.test(constant, "<=", 10) * b.test(constant, ">=", 0)
            b.test("x", "<=", "c")
            for i in range(n):
                constant = "c{}".format(i + 1)
                #b.ints(constant)
                test_dict[i+1] = b.test("x", "<=", random.randint(10, 90) )

            leaf_1 = b.exp(3)
            leaf_2 = b.exp(11)

            path_1 = leaf_1
            path_2 = leaf_2
            for i in range(n):
                index = n - i
                constant = "c{}".format(index)
                current_test = test_dict[index] # b.test("x", "<=", constant)
                path_1_old, path_2_old = path_1, path_2
                path_1 = b.ite(current_test, path_1_old, path_2_old)
                path_2 = b.ite(current_test, path_2_old, path_1_old)

            return bounds * b.ite(b.test("x", "<=", "c"), path_1, path_2)
        import math
        for size in range(5, 15):
            print("Testing XOR for n={}".format(size))
	    #self.compare_results(build_xor(size)[0], "c1")
	    #var = "c{}".format(size)
            #print(var)
	    #self.compare_results(build_xor(size)[0], var)
	    #continue
            #self.compare_results(build_xor(size)[0], "c{}".format(int(math.ceil((size+1)/2))))
            stop_watch = timer.Timer()
            xor = build_xor(size)
            #res = xor_id
            #vars_1 = list((["c{}".format(i + 1) for i in range(size)]))
            #vars_1 = reversed(["x", "c"] + list(reversed(["c{}".format(i + 1) for i in range(size-1)]))) 
            #vars_1 = reversed(vars_1)
            #for var_name in vars_1:
	    #	print(var_name)
	    #    res = self.compare_results(res, var_name)

	    #continue
	    resolve = bounds_diagram.BoundResolve(xor.pool, cache_result=True)
            reducer = LinearReduction(xor.pool)
	    result_id = xor.root_id
            control_id = xor.root_id
#
            vars_1 = list((["c{}".format(i + 1) for i in range(size)]))
            #vars_1 = reversed(["x", "c"] + list(reversed(["c{}".format(i + 1) for i in range(size-1)]))) 
            vars_1 = reversed(vars_1)
            vars_1 = list(vars_1)# + ["c", "x"]
	    vars_1 = ["c", "x"]
            stop_watch.start("bound_resolve:")
            for var_name in vars_1:
                #print(var_name)
                result_id = resolve.integrate(result_id, var_name)
                #result_id = reducer.reduce(result_id)
            stop_watch.stop()
            stop_watch.start("path enumerator:")
            for var_name in vars_1:
                #print(var_name)
                #print(var_name)
                control_id = matrix_vector.sum_out(xor.pool, control_id, [var_name])
                #control_id = reducer.reduce(control_id)
            stop_watch.stop()
            #self.compare_results(build_xor(size), "c{}".format(int(math.ceil(size/2))))

    def compare_recursively(self, test_diagram, var):
        root_node = test_diagram.root_node
        print("Comparing node {}".format(root_node))
        if root_node.is_terminal():
            return
        else:
            self.compare_recursively(test_diagram.pool.diagram(root_node.child_true), var)
            self.compare_recursively(test_diagram.pool.diagram(root_node.child_false), var)

            resolve = bounds_diagram.BoundResolve(test_diagram.pool)
            result_id_resolve = resolve.integrate(test_diagram.root_id, var)
            result_id_control = matrix_vector.sum_out(test_diagram.pool, test_diagram.root_id, [var])

            result_resolve = test_diagram.pool.diagram(result_id_resolve).evaluate({})
            result_control = test_diagram.pool.diagram(result_id_control).evaluate({})

            if not math.isinf(result_control):
                try:
                    self.assertEquals(result_control, result_resolve)
                except AssertionError as e:
                    print("Failure at node {}".format(root_node.node_id))
                    print("Resolve gives {} while control gives {}".format(result_resolve, result_control))
                    self.resolve_export.export(test_diagram, "failed_diagram")
                    raise e
                print("Match at node {}: {}".format(root_node.node_id, result_control))
            else:
                print("Result at node {} is infinite ({} for bound resolve)".format(root_node.node_id, result_resolve))

    def compare_results_paths(self, test_diagram, var):
        paths = self.get_paths(test_diagram)

        pool = test_diagram.pool
        resolve = bounds_diagram.BoundResolve(pool)

        def compare(path_to_compare):
            result_id_resolve = resolve.integrate(path_to_compare.root_id, var)
            result_id_control = matrix_vector.sum_out(path_to_compare.pool, path_to_compare.root_id, [var])

            result_resolve = pool.diagram(result_id_resolve).evaluate({})
            result_control = pool.diagram(result_id_control).evaluate({})

            if not math.isinf(result_control):
                try:
                    self.assertAlmostEquals(result_control, result_resolve, delta=10 ** -3)
                except AssertionError as e:
                    print("Resolve at {} gives {} while control gives {}".format(path_to_compare.root_id,
                                                                                 result_resolve, result_control))
                    self.resolve_export.export(path_to_compare, "failed_diagram")
                    raise e
                print("Match at {}: {}".format(path_to_compare.root_id, result_control))
            else:
                print("Path at {} is infinite ({} for bound resolve)".format(path_to_compare.root_id, result_resolve))

        for i in range(len(paths)):
            print("Trying path {} of {}".format(i, len(paths)))
            path = paths[i]
            compare(path)

        print("Summing")
        summed = pool.diagram(pool.terminal(0))  # path_67, path_55
        for i in range(len(paths)):
            path = paths[i]
            summed = summed + path
            print("Trying summed path {} of {}".format(i, len(paths)))
            compare(summed)

    def get_paths(self, test_diagram, path_diagram=None):
        b = Builder(test_diagram.pool)
        root_node = test_diagram.root_node
        if path_diagram is None:
            path_diagram = b.exp(1)
        if root_node.is_terminal():
            full_path_diagram = path_diagram * b.exp(root_node.expression)
            return [full_path_diagram]
        else:
            test_true = LinearTest(root_node.test.operator)
            path_diagram_true = path_diagram * test_diagram.pool.diagram(test_diagram.pool.bool_test(test_true))

            test_false = LinearTest(~root_node.test.operator)
            path_diagram_false = path_diagram * test_diagram.pool.diagram(test_diagram.pool.bool_test(test_false))

            paths_true = self.get_paths(test_diagram.pool.diagram(root_node.child_true), path_diagram_true)
            paths_false = self.get_paths(test_diagram.pool.diagram(root_node.child_false), path_diagram_false)
            return paths_true + paths_false

    def compare_results(self, test_diagram, var, result=None):
        import os
        from tests import export
        from pyxadd import bounds_diagram

        exporter = export.Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "resolve", True)
        stop_watch = timer.Timer()
        reducer = LinearReduction(test_diagram.pool)

        exporter.export(test_diagram, "test_diagram")

        stop_watch.start("Integrating using path enumeration")
        control_id = matrix_vector.sum_out(test_diagram.pool, test_diagram.root_id, [var])
        stop_watch.stop()

        control_diagram = test_diagram.pool.diagram(control_id)
        exporter.export(control_diagram, "path_enum_result")

        reduced_control = test_diagram.pool.diagram(reducer.reduce(control_id))
        exporter.export(reduced_control, "path_enum_result_reduced")

        resolve = bounds_diagram.BoundResolve(test_diagram.pool, cache_result=True)#, "./visual/resolve/debug/")

        stop_watch.start("Integrating using bound resolve")
        result_id = resolve.integrate(test_diagram.root_id, var)
        stop_watch.stop()

        result_diagram = test_diagram.pool.diagram(result_id)
        exporter.export(result_diagram, "bound_resolve_result")
        reduced_result = test_diagram.pool.diagram(reducer.reduce(result_id))
        exporter.export(reduced_result, "bound_resolve_result_reduced")
	return reduced_result
        #bounds_resolve_result = reduced_result.evaluate({})
        #path_enumeration_result = reduced_control.evaluate({})
        #if result is not None:
        #    self.assertAlmostEquals(result, bounds_resolve_result, delta=10**-3)
        #    self.assertAlmostEquals(result, path_enumeration_result, delta=10 ** -3)
        #self.assertAlmostEquals(bounds_resolve_result, path_enumeration_result,
        #                        msg="Result ({}) and control ({}) do not agree"
        #                       .format(bounds_resolve_result, path_enumeration_result), delta=10 ** -3)

    def test_multiplication(self):
        pool = Pool()
        pool.int_var("x1", "x2")
        x_two = Diagram(pool, pool.terminal("x2"))
        two = Diagram(pool, pool.terminal("2"))
        three = Diagram(pool, pool.terminal("3"))
        four = Diagram(pool, pool.terminal("4"))

        test11 = Diagram(pool, pool.bool_test(LinearTest("x1", ">=")))
        test12 = Diagram(pool, pool.bool_test(LinearTest("x1 - 1", "<=")))
        test13 = Diagram(pool, pool.bool_test(LinearTest("x1 - 3", ">")))

        test21 = Diagram(pool, pool.bool_test(LinearTest("x2", ">=")))
        test22 = Diagram(pool, pool.bool_test(LinearTest("x2", ">")))
        test23 = Diagram(pool, pool.bool_test(LinearTest("x2 - 1", ">")))
        test24 = Diagram(pool, pool.bool_test(LinearTest("x2 - 2", ">")))

        x_twos = test12 * ~test23 * x_two
        twos = test12 * test23 * two
        threes = ~test12 * ~test22 * three
        fours = ~test12 * test22 * four

        unlimited = x_twos + twos + threes + fours
        restricted = unlimited * test11 * ~test13 * test21 * ~test24

        vector = test21 * ~test24 * Diagram(pool, pool.terminal("x2 + 1"))

        result = Diagram(pool, matrix_multiply(pool, restricted.root_node.node_id, vector.root_node.node_id, ["x2"]))
        for x1 in range(0, 4):
            self.assertEqual(8 if x1 < 2 else 23, result.evaluate({"x1": x1}))


if __name__ == '__main__':
    unittest.main()
