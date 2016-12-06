from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix
from pyxadd.view import export

lb = 1
ub = 10


def solve(A, b, w, m=1000, delta=10**-3):
    norm_b = b.norm()
    x = b
    print("Norm of b is {}".format(norm_b))

    # export(x, "visual/richardson/x.dot")
    # export(A, "visual/richardson/A.dot")
    for i in range(m):
        renamed = x.rename({"r": "c"})
        renamed.export("visual/richardson/renamed{}.dot".format(i))
        A.element_product(renamed).export("visual/richardson/Aex{}.dot".format(i))
        A_times_x = A * renamed
        A_times_x.export("visual/richardson/Ax{}_full.dot".format(i))
        A_times_x = A_times_x.reduce()
        A_times_x.export("visual/richardson/Ax{}.dot".format(i))
        diff_to_b = b - A_times_x
        norm_difference = diff_to_b.norm()
        residual = float(norm_difference) / norm_b
        print("Residual is {} / {} = {}".format(norm_difference, norm_b, residual))
        if residual < delta:
            print("Found solution: ", [x.evaluate({"r": i}) for i in range(lb, ub + 1)])
            return x

        x = x + w * diff_to_b
        x.export("visual/richardson/x_full{}.dot".format(i))
        x = x.reduce()
        x.export("visual/richardson/x{}.dot".format(i))
        print("X: ", [x.evaluate({"r": i}) for i in range(lb, ub + 1)])
        exit()

pool = Pool()
build = Builder(pool)
build.ints("r", "c")

bounds = build.limit("r", lb, ub) & build.limit("c", lb, ub)

A_d = bounds \
      & build.test("r", "<=", "c") \
      & build.test("r", ">=", "c") * build.exp(2)

A_db = bounds & (build.test("r", ">", lb + (ub - lb) / 2) & build.test("c", "<=", lb + (ub - lb) / 2)
                 | (build.test("r", "<=", lb + (ub - lb) / 2) & build.test("c", ">", lb + (ub - lb) / 2)))

A = Matrix(A_d * build.terminal(3) + A_db, [("r", lb, ub)], [("c", lb, ub)], auto_reduce=False)
# TODO WHAT IS THAT 7
A.print_ground()
A.reduce()
A.export("visual/richardson/A.dot")

b_d = build.limit("r", lb, ub) * build.exp(10)
b = Matrix(b_d, [("r", lb, ub)], [], auto_reduce=False)
b.transpose().print_ground()
b.export("visual/richardson/b.dot")

result = solve(A, b, 0.05)
