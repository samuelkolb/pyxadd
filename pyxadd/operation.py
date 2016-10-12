class Operation(object):
    def __init__(self, symbol):
        self._symbol = symbol

    @property
    def symbol(self):
        return self._symbol

    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # deal with NaN?
        # special cases
        # two terminals
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.symbol)


class Multiplication(Operation):
    def __init__(self):
        Operation.__init__(self, "*")

    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from pyxadd.diagram import TerminalNode
        if node1.node_id == pool.zero_id or node2.node_id == pool.zero_id:
            return pool.zero_id
        elif node1.node_id == pool.one_id:
            return node2.node_id
        elif node2.node_id == pool.one_id:
            return node1.node_id
        elif isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode):
            return pool.terminal(node1.expression * node2.expression)
        return None


class Summation(Operation):
    def __init__(self):
        Operation.__init__(self, "+")

    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from pyxadd.diagram import TerminalNode
        if node1.node_id == pool.zero_id:
            return node2.node_id
        elif node2.node_id == pool.zero_id:
            return node1.node_id
        elif node1.node_id == pool.pos_inf_id or node1.node_id == pool.neg_inf_id:
            return node1.node_id
        elif node2.node_id == pool.pos_inf_id or node2.node_id == pool.neg_inf_id:
            return node2.node_id
        elif isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode):
            return pool.terminal(node1.expression + node2.expression)
        return None


class LogicalOr(Operation):
    def __init__(self):
        Operation.__init__(self, "|")

    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from pyxadd.diagram import TerminalNode
        if isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode) \
                and ((node1.node_id != pool.zero_id and node1.node_id != pool.one_id)
                     or (node2.node_id != pool.zero_id and node2.node_id != pool.one_id)):
            raise RuntimeError("Nodes must be one or zero")

        if node1.node_id == pool.zero_id:
            return node2.node_id
        elif node1.node_id == pool.one_id:
            return pool.one_id
        elif node2.node_id == pool.zero_id:
            return node1.node_id
        elif node2.node_id == pool.one_id:
            return pool.one_id
        elif isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode):
            raise RuntimeError("Cases should be covered")
        return None


class LogicalAnd(Operation):
    def __init__(self):
        Operation.__init__(self, "&")

    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from pyxadd.diagram import TerminalNode
        if isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode) \
                and ((node1.node_id != pool.zero_id and node1.node_id != pool.one_id)
                     or (node2.node_id != pool.zero_id and node2.node_id != pool.one_id)):
            raise RuntimeError("Nodes must be one or zero, were {} and {}".format(node1, node2))

        if node1.node_id == pool.one_id:
            return node2.node_id
        elif node1.node_id == pool.zero_id:
            return pool.zero_id
        elif node2.node_id == pool.one_id:
            return node1.node_id
        elif node2.node_id == pool.zero_id:
            return pool.zero_id
        elif isinstance(node1, TerminalNode) and isinstance(node2, TerminalNode):
            raise RuntimeError("Cases should be covered")
        return None
