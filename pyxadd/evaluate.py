import numpy

from pyxadd.walk import DepthFirstWalker


class EvaluationWalker(DepthFirstWalker):
    def __init__(self, diagram, entries):
        DepthFirstWalker.__init__(self, diagram)
        self.entries = [{str(k): v for k, v in entry.items()} for entry in entries]
        self.results = numpy.zeros(len(entries))

    def _initial(self):
        return set(range(len(self.entries)))

    def visit_terminal(self, terminal_node, parent_message):
        if parent_message is None:
            parent_message = self._initial()

        for i in parent_message:
            self.results[i] = terminal_node.evaluate(self.entries[i])

    def visit_internal(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = self._initial()

        true_message = set()
        false_message = set()

        for i in parent_message:
            if internal_node.test.evaluate(self.entries[i]):
                true_message.add(i)
            else:
                false_message.add(i)

        return true_message, false_message


def mass_evaluate(diagram, assignments):
    walker = EvaluationWalker(diagram, assignments)
    walker.walk()
    return walker.results

