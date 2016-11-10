from __future__ import print_function

import itertools
from png import Writer

from pyxadd.build import Builder
from pyxadd.diagram import Pool, Diagram
from pyxadd.matrix.matrix import assignments
from pyxadd.reduce import SmtReduce, LinearReduction
from pyxadd.view import export

pool = Pool()
pool.add_var("x", "int")
pool.add_var("xs", "int")
pool.add_var("y", "int")
pool.add_var("ys", "int")
b = Builder(pool)

# TODO needs control over interleaving


class RedGreenBlueDiagrams(object):
    def __init__(self, red, green, blue):
        self.diagrams = (red, green, blue)

    @staticmethod
    def all(diagram):
        return RedGreenBlueDiagrams(diagram, diagram, diagram)

    def _binary(self, op, other):
        if isinstance(other, RedGreenBlueDiagrams):
            return RedGreenBlueDiagrams(*(op(self.diagrams[i], other.diagrams[i]) for i in range(3)))
        elif isinstance(other, Diagram):
            return RedGreenBlueDiagrams(*(op(self.diagrams[i], other) for i in range(3)))
        elif isinstance(other, tuple):
            return RedGreenBlueDiagrams(*(op(self.diagrams[i], other[i])
                                          if other[i] is not None else self.diagrams[i]
                                          for i in range(3)))
        elif isinstance(other, dict):
            return RedGreenBlueDiagrams(*(op(self.diagrams[i], other[c])
                                          if c in other else self.diagrams[i]
                                          for i, c in enumerate(["red", "green", "blue"])))

    def __add__(self, other):
        return self._binary(lambda d1, d2: d1 + d2, other)

    def __mul__(self, other):
        return self._binary(lambda d1, d2: d1 * d2, other)

    def __sub__(self, other):
        return self._binary(lambda d1, d2: d1 - d2, other)

    def __invert__(self):
        return RedGreenBlueDiagrams(*(~self.diagrams[i] for i in range(3)))

    def __or__(self, other):
        return self._binary(lambda d1, d2: d1 | d2, other)

    def __and__(self, other):
        return self._binary(lambda d1, d2: d1 & d2, other)

screen_width = 240
screen_height = 120

# FIXME: RGB : Extra variable!!

grid_size = 5
grid_domain_size = grid_size * 2

column_variables = [("x", 0, screen_width / grid_domain_size - 1), ("xs", 0, grid_domain_size - 1)]
row_variables = [("y", 0, screen_height / grid_domain_size - 1), ("ys", 0, grid_domain_size - 1)]


def reduce(diagram, linear=True):
    variables = [t[0] for t in column_variables + row_variables]
    if linear:
        reduction = LinearReduction(diagram.pool)
        result = reduction.reduce(diagram.root_node.node_id, variables)
    else:
        reduction = SmtReduce(diagram.pool)
        result = reduction.reduce(diagram.root_node.node_id, variables)
    return Diagram(diagram.pool, result)


screen_x = "({} * x + xs)".format(grid_domain_size)
screen_y = "({} * y + ys)".format(grid_domain_size)

# Build diagram with vertical and horizontal stripes

x_stripe_on = b.test("xs", ">", grid_size - 1)
y_stripe_on = b.test("ys", ">", grid_size - 1)

grid = x_stripe_on | y_stripe_on

# Build object to be displayed
# screen_x = 2 * x + xs, screen_y = 2 * y + ys

border = 5
available_width = screen_width - 2 * border
available_height = screen_height - 2 * border
center_padding = int(available_height / 4)

x_start_1 = border
x_start_2 = border + available_width / 3 - 1
x_start_3 = border + 2 * available_width / 3 - 1

center_y = screen_height / 2


def slope_intercept(p1, p2):
    #   (I) y1 = a * x1 + b
    #  (Ia) b = y1 - a * x1
    #  (II) y2 = a * x2 + b
    # (IIa) y2 = a * x2 + y1 - a * x1
    # (IIb) a = (y2 - y1) / (x2 - x1)
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / float(x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


left = b.limit(screen_x, x_start_1, x_start_2 - 1) \
       & b.limit(screen_y, border, screen_height - border - 1)

slope_1, intercept_1 = slope_intercept((border, center_y), (x_start_2 - 1, border))
left &= b.test("{y}".format(y=screen_y), ">=", "{a} * {x} + {b}".format(a=slope_1, x=screen_x, b=intercept_1))

slope_2, intercept_2 = slope_intercept((border, center_y), (x_start_2 - 1, screen_height - border))
left &= b.test("{y}".format(y=screen_y), "<=", "{a} * {x} + {b}".format(a=slope_2, x=screen_x, b=intercept_2))

center = b.limit(screen_x, x_start_2, x_start_3 - 1)\
         & b.limit(screen_y, border + center_padding, screen_height - border - center_padding - 1)

right = b.limit(screen_x, x_start_3, screen_width - border - 1)\
        & b.limit(screen_y, border, screen_height - border - 1)

slope_1, intercept_1 = slope_intercept((screen_width - border, center_y), (x_start_3, border))
right &= b.test("{y}".format(y=screen_y), ">=", "{a} * {x} + {b}".format(a=slope_1, x=screen_x, b=intercept_1))

slope_2, intercept_2 = slope_intercept((screen_width - border, center_y), (x_start_3, screen_height - border))
right &= b.test("{y}".format(y=screen_y), "<=", "{a} * {x} + {b}".format(a=slope_2, x=screen_x, b=intercept_2))


full_object = left | center | right
color = b.terminal(170)
export(full_object, "visual/full_object.dot")

# Build light

# intensity = 1 / (screen_x ^ 2 + screen_y ^ 2)
light = b.terminal("1 / (1 + (({xc} - {x}) ** 2 + ({yc} - {y}) ** 2) * 0.0001)"
                   .format(xc=screen_width / 2, x=screen_x, yc=screen_height / 2, y=screen_y))

# Combine grid and object

zero = b.terminal(0)
one = b.terminal(1)
full = b.terminal(255)
grid_discount = b.terminal(0.6)
light_factor = b.terminal(1)
background = b.terminal(100)
combined = b.ite(full_object, color, background)
light_effect = full_object * ((full - combined) * light) * light_factor

# combined = RedGreenBlueDiagrams.all(combined, combined, combined)
# combined += {"green": light_effect}

combined += light_effect

grid_overlay = combined * grid * grid_discount
# grid_overlay -= grid_overlay * light * b.terminal(0.6)
combined -= grid_overlay

# combined = reduce(combined)

if isinstance(combined, Diagram):
    export(combined, "visual/combined.dot")
    print("Exported combined diagram")
    combined_reduced = reduce(combined)
    export(combined_reduced, "visual/combined_reduced.dot")
    print("Exported reduced combined diagram")
    export(grid, "visual/grid.dot")
    print("Exported grid diagram")
    export(light, "visual/light.dot")
    print("Exported light diagram")


# Draw png

def get_rows(diagram, column_variables, row_variables, transform=None):
    for row_assignment in assignments(row_variables):
        row = []
        for assignment in assignments(column_variables, fixed=row_assignment):
            if isinstance(diagram, Diagram):
                value = diagram.evaluate(assignment)
                if transform is not None:
                    value = transform(value)
                row.append(value)
            elif isinstance(diagram, RedGreenBlueDiagrams):
                for d in diagram.diagrams:
                    value = d.evaluate(assignment)
                    if transform is not None:
                        value = transform(value)
                    row.append(value)
        yield row


def draw(name, diagram, column_variables, row_variables):
    rows = list(get_rows(diagram, column_variables, row_variables, transform=int))
    writer = Writer(width=screen_width, height=screen_height, greyscale=isinstance(diagram, Diagram))
    with open("visual/{}.png".format(name), "w") as outfile:
        writer.write(outfile, rows)


def print_ascii(diagram, column_variables, row_variables):
    for row in get_rows([diagram], column_variables, row_variables):
        print(*["{: 4.0f}".format(v) for v in row], sep="")


draw("combined", combined, column_variables, row_variables)
print("Drew combined")
draw("grid", grid * full, column_variables, row_variables)
print("Drew grid")
draw("light", light * full, column_variables, row_variables)
print("Drew light")
# print_ascii(combined, column_variables, row_variables)
