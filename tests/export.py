from pyxadd.diagram import Diagram
from pyxadd.matrix.matrix import Matrix
from pyxadd.view import export
import os


class Exporter(object):
    def __init__(self, base_dir, feature):
        self.base_dir = base_dir
        self.feature = feature

    def path(self):
        return os.path.join(self.base_dir, self.feature)

    def export(self, diagram, name):
        if isinstance(diagram, Diagram):
            export(diagram, os.path.join(self.path(), name + ".dot"))
        elif isinstance(diagram, Matrix):
            diagram.export(os.path.join(self.path(), name))
