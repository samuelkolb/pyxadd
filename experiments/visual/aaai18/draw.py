from pyxadd.bounds_diagram import BoundResolve
from pyxadd.build import Builder


def get_diagram():
    b = Builder()
    b.ints("x", "y", "a", "b")

    x0 = b.test("x >= 0")
    x1000 = b.test("x <= 1000")
    ab = b.test("a <= b")
    xa = b.test("x <= a")
    xb = b.test("x <= b")

    exp1 = b.exp("2 * x")

    return x0 * x1000 * b.ite(ab, xa * exp1, xb * exp1)


def get_x_a_diagram():
    b = Builder()
    b.ints("x", "y", "z")
    b.vars("bool", "a")

    xz = b.test("x <= z")
    a = b.test("a")
    x500 = b.test("x >= y")
    x1000 = b.test("x <= 1000")
    x0 = b.test("x >= 0")

    exp1 = b.exp("2 * x")

    return xz * ((x500 & x1000) | (~a & ~x500 & x0)) * exp1


def get_x_a_real_diagram():
    b = Builder()
    b.ints("x", "a")

    a = b.test("x <= a")
    x500 = b.test("x >= 500")
    x1000 = b.test("x <= 1000")
    x0 = b.test("x >= 0")

    exp1 = b.exp("2 * x")

    return ((x500 & x1000) | (~a & ~x500 & x0)) * exp1


def get_pentagon():
    b = Builder()
    b.ints("x", "y", "z")

    xy = b.test("x <= y")
    x10 = b.test("x <= z")
    x0 = b.test("x >= 0")
    exp1 = b.exp("2 * x")

    return (xy * x0 * x10 * exp1) + (~xy * x0 * x10 * exp1) + (~xy * ~x0 * exp1)


def get_example():
    b = Builder()
    b.ints("x", "y", "z")
    x0 = b.test("x >= 0")
    xa = b.test("x <= y")
    x10 = b.test("x <= z")
    exp1 = b.exp("2 * x")
    exp2 = b.exp("4 * x")

    return x0 * b.ite(xa, b.ite(x10, exp1, exp2), x10 * exp1)



def xor(d1, d2):
    return (d1 | d2) & ~(d1 & d2)


def get_xor_diagram():
    b = Builder()
    b.ints("x", "c1", "c2", "c3", "c4")

    x_c1 = b.test("x < c1")
    x_c2 = b.test("x < c2 + c1")
    x_c3 = b.test("x < c3 + c2")
    x_c4 = b.test("x < c4 + c3")

    return xor(xor(xor(x_c1, x_c2), x_c3), x_c4)

def get_paper_illustration():
    b = Builder()
    b.ints("x", "y", "z", "v", "w")

    t1 = b.test("x >= v")
    t2 = b.test("x >= y")
    t3 = b.test("x <= z")
    t4 = b.test("x >= w")

    e1 = b.exp("3 * x * x")
    e2 = b.exp("2 * x")

    # \theta = x \geq v \land (x < y \lor x \leq z)$ and $w = \ite(x \leq z \land x \geq w, 3x^2, 2x).
    return t1 & (~t2 | t3) * b.ite(t3 & t4, e1, e2)


if __name__ == "__main__":
    d = get_paper_illustration()
    d.export("paper_illustration.dot", True)

    # d = get_example().reduce(method="smt")
    # d.export("example.dot", True)
    # resolve = BoundResolve(d.pool)
    # d.pool.diagram(resolve.integrate(d.root_id, "x")).reduce(method="smt").export("integrated", True)
    # print("{} cache hits".format(resolve.cache_hits))

    # d = get_xor_diagram()
    # d.export("xor", True)
    # resolve = BoundResolve(d.pool)
    # d.pool.diagram(resolve.integrate(d.root_id, "c1")).reduce(method="simple").export("xor_integrated", True)
    # print("{} cache hits".format(resolve.cache_hits))
