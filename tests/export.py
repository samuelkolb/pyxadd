from pyxadd.diagram import Diagram
from pyxadd.matrix.matrix import Matrix
from pyxadd.view import export
import os


class Exporter(object):
    def __init__(self, base_dir, feature, print_node_ids=False):
        self.base_dir = base_dir
        self.feature = feature
        self.print_node_ids = print_node_ids

    def path(self):
        return os.path.join(self.base_dir, self.feature)

    def export(self, diagram, name, print_node_ids=None):
        if print_node_ids is None:
            print_node_ids = self.print_node_ids
        if isinstance(diagram, Diagram):
            export(diagram, os.path.join(self.path(), name + ".dot"), print_node_ids=print_node_ids)
        elif isinstance(diagram, Matrix):
            diagram.export(os.path.join(self.path(), name))
