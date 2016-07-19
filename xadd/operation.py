class Operation:
    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # deal with NaN?
        # special cases
        # two terminals
        raise NotImplementedError()


class Multiplication(Operation):
    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from xadd.diagram import TerminalNode
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
    @classmethod
    def compute_terminal(cls, pool, node1, node2):
        # TODO deal with NaN?
        from xadd.diagram import TerminalNode
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
