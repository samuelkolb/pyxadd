from pyxadd.diagram import InternalNode, TerminalNode


class Walker:
    def __init__(self, diagram):
        self._diagram = diagram

    def walk(self):
        """
        Walk the given diagram.
        :return: The result
        """
        raise NotImplementedError()


class DepthFirstWalker(Walker):
    def visit_terminal(self, terminal_node, parent_message):
        """
        Visit a terminal node.
        :param terminal_node: The terminal node to visit
        :param parent_message: The message received from the parent
        :return: The result message to be passed up
        """
        raise NotImplementedError()

    def visit_internal(self, internal_node, parent_message):
        """
        Visit an internal node.
        :param internal_node: The internal node to visit
        :param parent_message: The message received from the parent (is None for root)
        :return: A tuple of messages to be passed to the true and false child node respectively
        """
        raise NotImplementedError()

    def walk(self):
        """
        Walks the diagram without computing a result.
        """
        self._visit(self._diagram.root_node)

    def _visit(self, node, message=None):
        if isinstance(node, TerminalNode):
            self.visit_terminal(node, message)
        elif isinstance(node, InternalNode):
            true_message, false_message = self.visit_internal(node, message)
            self._visit(self._diagram.node(node.child_true), true_message)
            self._visit(self._diagram.node(node.child_false), false_message)
        else:
            raise RuntimeError("Unexpected node type {}.".format(type(node)))


class DownUpWalker(Walker):
    def visit_terminal(self, terminal_node, parent_message):
        """
        Visit a terminal node.
        :param terminal_node: The terminal node to visit
        :param parent_message: The message received from the parent
        :return: The result message to be passed up
        """
        raise NotImplementedError()

    def visit_internal_down(self, internal_node, parent_message):
        """
        Visit an internal node on the way down.
        :param internal_node: The internal node to visit
        :param parent_message: The message received from the parent
        :return: A tuple of messages to be passed to the true and false child node respectively
        """
        raise NotImplementedError()

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        """
        Visit an internal node on the way up.
        :param internal_node: The internal node to visit
        :param true_result: The message received from the true child node
        :param false_result: The message received from the false child node
        :return: The message to be passed up
        """
        raise NotImplementedError()

    def walk(self):
        return self._visit(self._diagram.root_node)

    def _visit(self, node, message=None):
        if isinstance(node, TerminalNode):
            return self.visit_terminal(node, message)
        elif isinstance(node, InternalNode):
            true_message, false_message = self.visit_internal_down(node, message)
            true_result = self._visit(self._diagram.node(node.child_true), true_message)
            false_result = self._visit(self._diagram.node(node.child_false), false_message)
            return self.visit_internal_aggregate(node, true_result, false_result)
        else:
            raise RuntimeError("Unexpected node type {}.".format(type(node)))


class ParentsWalker(DepthFirstWalker):
    def __init__(self, diagram):
        DepthFirstWalker.__init__(self, diagram)
        self._nodes = None

    def walk(self):
        self._nodes = {self._diagram.root_node.node_id: set()}
        DepthFirstWalker.walk(self)
        nodes = self._nodes
        self._nodes = None
        return nodes

    def visit_internal(self, internal_node, parent_message):
        self._update_parents(internal_node, parent_message)
        return internal_node.node_id, internal_node.node_id

    def visit_terminal(self, terminal_node, parent_message):
        self._update_parents(terminal_node, parent_message)

    def _update_parents(self, internal_node, parent):
        if parent is not None:
            if internal_node.node_id not in self._nodes:
                self._nodes[internal_node.node_id] = set()
            self._nodes[internal_node.node_id].add(parent)


class WalkingProfile:
    def __init__(self, diagram):
        parents = ParentsWalker(diagram).walk()
        counts = {n: len(parents[n]) for n in parents}
        nodes_and_counts = list((n, counts[n]) for n in WalkingProfile.extract_cache(parents, diagram))
        self._nodes = list(n for n, _ in nodes_and_counts)
        self._counts = {n: (c, 0) for n, c in nodes_and_counts}
        self._next = 0

    def reset(self):
        self._counts = {n: (self._counts[n][0], 0) for n in self._counts}
        self._next = 0

    def count(self, node):
        c, i = self._counts[node]
        i += 1
        self._counts[node] = (c, i)
        if i == c:
            return True
        elif i < c:
            return False
        else:
            raise RuntimeError("Count is already saturated")

    def has_next(self):
        return self._next < len(self._nodes)

    def next(self):
        current = self._nodes[self._next]
        self._next += 1
        return current

    @staticmethod
    def extract_cache(parents, diagram):
        nested_reverse = WalkingProfile.extract_layers(diagram, parents)
        reverse_list = []
        for i in range(0, len(nested_reverse)):
            for node_id in nested_reverse[i]:
                reverse_list.append(node_id)
        return reversed(reverse_list)

    @staticmethod
    def extract_layers(diagram, parents):
        root_id = diagram.root_node.node_id
        positions = {root_id: 0}
        watch = [root_id]
        while len(watch) > 0:
            current_id = watch.pop()
            current_parents = parents[current_id]
            if not len(current_parents) == 0:
                raise RuntimeError("Parents not empty, found {}.".format(current_parents))
            current_node = diagram.node(current_id)
            if isinstance(current_node, InternalNode):
                for child_id in (current_node.child_true, current_node.child_false):
                    if child_id not in positions:
                        positions[child_id] = 0
                    parents[child_id].remove(current_id)
                    positions[child_id] = max(positions[child_id], positions[current_id] + 1)
                    if len(parents[child_id]) == 0:
                        watch.append(child_id)
        nested_reverse = dict()
        for node_id in positions:
            count = positions[node_id]
            if count not in nested_reverse:
                nested_reverse[count] = []
            nested_reverse[count].append(node_id)
        return nested_reverse


class BottomUpWalker(Walker):
    def __init__(self, diagram, profile):
        Walker.__init__(self, diagram)
        self._profile = profile

    def visit_terminal(self, terminal_node):
        """
        Visit a bottom terminal node.
        :param terminal_node: The terminal node to visit
        :return: The message to be passed up
        """
        raise NotImplementedError()

    def visit_internal(self, internal_node, true_message, false_message):
        """
        Visit an internal node.
        :param internal_node: The internal node to visit
        :param true_message: The message received from the true child node
        :param false_message: The message received from the false child node
        :return: The message to be passed up
        """
        raise NotImplementedError()

    def walk(self):
        messages = dict()
        while self._profile.has_next():
            node = self._diagram.node(self._profile.next())
            if isinstance(node, TerminalNode):
                messages[node.node_id] = self.visit_terminal(node)
            elif isinstance(node, InternalNode):
                true_message = self._retrieve_message(node.child_true, messages)
                false_message = self._retrieve_message(node.child_false, messages)
                messages[node.node_id] = self.visit_internal(node, true_message, false_message)
            else:
                raise RuntimeError("Unexpected node type {}.".format(type(node)))
        if len(messages) != 1:
            raise RuntimeError("Message cache not reduced to 1.")
        root, result = messages.popitem()
        if root != self._diagram.root_node.node_id:
            raise RuntimeError("Remaining node not root.")
        return result

    def _retrieve_message(self, node_id, messages):
        if self._profile.count(node_id):
            return messages.pop(node_id, None)
        else:
            return messages[node_id]
