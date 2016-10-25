import colorsys
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
from pyxadd.diagram import InternalNode

def visualize(matrix, filename):
    mapping = dict()
    for x in range(1, matrix.width + 1):
        for y in range(1, matrix.height + 1):
            diagram = matrix.diagram
            node = find_node(x, y, diagram)
            mapping[x, y] = node.expression

    draw(mapping, matrix.width, matrix.height, filename)


def find_node(x, y, diagram):
    node = diagram.root_node
    while isinstance(node, InternalNode):
        if node.test.evaluate({'x': x,
         'y': y}):
            node = diagram.node(node.child_true)
        else:
            node = diagram.node(node.child_false)

    return node


def visualize_ground(matrix, filename):
    mapping = dict()
    for x in range(1, matrix.width + 1):
        for y in range(1, matrix.height + 1):
            diagram = matrix.diagram
            mapping[x, y] = diagram.evaluate({'x': x,
             'y': y})

    draw(mapping, matrix.width, matrix.height, filename)


def get_colors(n):
    hsv_tuples = [ (x * 1.0 / n, 0.75, 0.75) for x in range(n) ]
    hex_colors = []
    for hsv_tuple in hsv_tuples:
        rgb = colorsys.hsv_to_rgb(*hsv_tuple)
        hex_colors.append('#{0:02x}{1:02x}{2:02x}'.format(*[ int(x * 255) for x in rgb ]))

    return hex_colors


def draw(mapping, width, height, filename):
    labels = list(set(mapping.values()))
    try:
        import random
        random.shuffle(labels)
    except TypeError:
        pass

    colors = get_colors(len(labels))
    color_map = OrderedDict(zip(labels, colors))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x, y in mapping:
        rect = patches.Rectangle((x, y), 1, 1, facecolor=color_map[mapping[x, y]])
        ax.add_patch(rect)

    ax.set_xlim([1, width + 1])
    indices = numpy.array([1, width])
    ax.set_xticks(indices + 0.5)
    ax.set_xticklabels(indices)
    ax.set_ylim([1, height + 1])
    indices = numpy.array([1, height])
    ax.set_yticks(indices + 0.5)
    ax.set_yticklabels(indices)
    ax.invert_yaxis()
    legend_patches = []
    for label, color in color_map.items():
        legend_patches.append(patches.Patch(color=color_map[label], label=label))

    lgd = plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(filename, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
