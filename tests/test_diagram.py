import unittest

from pyxadd import leaf_transform
from pyxadd.build import Builder
from pyxadd.diagram import Diagram, Pool
from pyxadd.operation import Multiplication, Summation
from pyxadd.order import is_ordered
from pyxadd.test import LinearTest
from pyxadd.timer import Timer
from pyxadd.view import export
from pyxadd.walk import WalkingProfile, ParentsWalker


class TestDiagram(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        pool.int_var("x")

        self.test1 = pool.bool_test(LinearTest("x", ">="))
        self.test2 = pool.bool_test(LinearTest("x + 2", ">"))
        self.test3 = pool.bool_test(LinearTest("x + 1", "<="))
        self.test4 = pool.bool_test(LinearTest("x - 5", "<="))
        self.x = pool.terminal("x")

        p1 = pool.apply(Multiplication, self.test1, self.test4)
        p2 = pool.apply(Multiplication, pool.invert(self.test1), self.test2)
        p3 = pool.apply(Multiplication, pool.apply(Multiplication, pool.apply(Multiplication,
                                                                              pool.invert(self.test1),
                                                                              pool.invert(self.test2)),
                                                   self.test3),
                        self.test4)

        result = pool.apply(Summation, pool.apply(Summation, p1, p2), p3)
        result = pool.apply(Multiplication, result, self.x)
        self.diagram = Diagram(pool, result)

    def test_evaluation(self):
        self.assertEqual(4, self.diagram.evaluate({"x": 4}))

    def test_multiplication(self):
        pool = Pool()
        pool.int_var("x")
        two = pool.terminal("2")
        x = pool.terminal("x")

        test1 = pool.bool_test(LinearTest("x", ">="))
        test2 = pool.apply(Multiplication, pool.bool_test(LinearTest("x - 5", "<=")), x)

        product = pool.apply(Multiplication, test1, test2)
        result = Diagram(pool, pool.apply(Multiplication, product, two))

        for i in range(0, 10):
            evaluated = result.evaluate({"x": i})
            if 0 <= i <= 5:
                self.assertEqual(2 * i, evaluated)
            else:
                self.assertEqual(0, evaluated)

    def test_not(self):
        pool = Pool()
        pool.int_var("x")
        dd_true = Diagram(pool, pool.bool_test(LinearTest("x", ">=")))
        dd_false = Diagram(pool, pool.invert(dd_true.root_node.node_id))

        for i in range(-5, 6):
            assignment = {"x": i}
            self.assertEqual((dd_true.evaluate(assignment) + 1) % 2, dd_false.evaluate(assignment))

    def test_construct(self):
        self.check_diagram(self.diagram, self.diagram.pool.zero_id, self.x)

    def test_summation(self):
        result = self.diagram + self.diagram
        self.check_diagram(result, result.pool.zero_id, result.pool.terminal("2*x"))

    def check_diagram(self, diagram, zero_term, x_term):
        pool = diagram.pool
        layers = WalkingProfile.extract_layers(diagram, ParentsWalker(diagram).walk())
        self.assertEqual(5, len(layers))

        self.assertEqual(1, len(layers[0]), layers[0])
        self.assertEqual(pool.get_node(self.test1).test, pool.get_node(layers[0][0]).test, layers[0])

        self.assertEqual(1, len(layers[1]), layers[1])
        self.assertEqual(pool.get_node(self.test2).test, pool.get_node(layers[1][0]).test, layers[1])

        self.assertEqual(1, len(layers[2]), layers[2])
        self.assertEqual(pool.get_node(self.test3).test, pool.get_node(layers[2][0]).test, layers[2])

        self.assertEqual(1, len(layers[3]), layers[3])
        self.assertEqual(pool.get_node(self.test4).test, pool.get_node(layers[3][0]).test, layers[3])

        self.assertEqual(2, len(layers[4]))
        self.assertTrue(zero_term in layers[4], layers[4])
        self.assertTrue(x_term in layers[4], layers[4])

    def test_printing(self):
        import json
        encoded = Pool.to_json(self.diagram.pool)
        representation = json.loads(encoded)
        reconstructed = Pool.from_json(encoded)
        re_encoded = Pool.to_json(reconstructed)
        new_representation = json.loads(re_encoded)
        self.assertEquals(representation, new_representation)

    def test_inversion(self):
        pool = Pool()
        build = Builder(pool)
        build.vars("bool", "a", "b")
        build.vars("int", "x")

        test1 = build.test("a")
        test2 = build.test("b")
        test3 = build.test("x", "<=", 5)

        node3 = build.ite(test3, 1, 0)
        diagram = build.ite(test1, build.ite(test2, node3, 1), node3)

        self.assertTrue(is_ordered(diagram))

        def inversion1(root_id):
            minus_one = pool.terminal("-1")
            return pool.apply(Multiplication, pool.apply(Summation, root_id, minus_one), minus_one)

        def transform(terminal_node, d):
            if terminal_node.expression == 1:
                return d.pool.zero_id
            elif terminal_node.expression == 0:
                return d.pool.one_id
            else:
                raise RuntimeError("Could not invert value {}".format(terminal_node.expression))

        def inversion2(root_id):
            to_invert = pool.diagram(root_id)
            profile = WalkingProfile(diagram)
            return leaf_transform.transform_leaves(transform, to_invert)

        iterations = 1000
        timer = Timer(precision=6)
        timer.start("Legacy inversion")
        for _ in range(iterations):
            inversion1(diagram.root_id)
        time_legacy = timer.stop()

        inverted1 = pool.diagram(inversion1(diagram.root_id))

        timer.start("New inversion")
        for _ in range(iterations):
            inversion2(diagram.root_id)
        time_new = timer.stop()

        inverted2 = pool.diagram(inversion2(diagram.root_id))

        for a in [True, False]:
            for b in [True, False]:
                for x in range(10):
                    assignment = {"a": a, "b": b, "x": x}
                    self.assertNotEqual(diagram.evaluate(assignment), inverted1.evaluate(assignment))
                    self.assertNotEqual(diagram.evaluate(assignment), inverted2.evaluate(assignment))

        self.assertTrue(time_legacy > time_new, "New inversion ({}) not faster than legacy implementation ({})"
                        .format(time_new, time_legacy))



if __name__ == '__main__':
    unittest.main()
