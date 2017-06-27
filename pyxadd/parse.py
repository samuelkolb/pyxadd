import re

from pyxadd import build
from pyxadd.test import Test, LinearTest, Operator


class Node(object):
    def __init__(self, name=None):
        self.name = name
        self.children = []

    def has_name(self):
        return self.name is not None

    def __repr__(self):
        return "Node({}, {})".format(self.name, self.children)


def string_to_ast(string, operators=None):
    return tokenized_string_to_ast(tokenize(string), operators)


def tokenize(chars):
    """Convert a string of characters into a list of tokens."""
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()


def tokenized_string_to_ast(tokenized_string, operators=None):
    stack = []
    root = None
    operators = set(operators) if operators is not None else None
    for token in tokenized_string:
        current = stack[-1] if len(stack) > 0 else None
        if token == "(":
            node = Node()
            stack.append(node)
            if current is not None:
                current.children.append(node)
            else:
                root = node
        elif token == ")":
            stack.pop()
        else:
            if current.has_name():
                current.children.append(Node(token))
            else:
                if operators is not None and token not in operators:
                    raise RuntimeError("Disallowed token '{}'".format(token))
                current.name = token
    return root


class XADDParser(object):
    operators = ["ite", "^", "~", "&", "|", "*", "+", "<=", "<", "const", "var"]

    def __init__(self, pool=None):
        self.builder = build.Builder(pool)

    def parse_xadd(self, nested_string):
        return self.ast_to_xadd(string_to_ast(nested_string, self.operators))

    def ast_to_xadd(self, node):
        """
        :type node: Node
        """

        def apply_op(operator):
            xadds = [self.ast_to_xadd(child) for child in node.children]
            result = xadds[0]
            for i in range(1, len(xadds)):
                result = operator(result, xadds[i])
            return result

        if node.name == "ite":
            return self.builder.ite(*[self.ast_to_xadd(node.children[i]) for i in range(3)])
        elif node.name == "~":
            return ~self.ast_to_xadd(node.children[0])
        elif node.name == "^":
            return self.builder.exp(self.ast_to_expression(node))
        elif node.name == "&":
            return apply_op(lambda x1, x2: x1 & x2)
        elif node.name == "|":
            return apply_op(lambda x1, x2: x1 | x2)
        elif node.name == "*":
            return apply_op(lambda x1, x2: x1 * x2)
        elif node.name == "+":
            return apply_op(lambda x1, x2: x1 + x2)
        elif node.name in ["<=", "<", ">=", ">"]:
            lhs = self.ast_to_expression(node.children[0])
            rhs = self.ast_to_expression(node.children[1])
            return self.builder.test(lhs, node.name, rhs)
        elif node.name == "const":
            return self._get_constant(node)
        elif node.name == "var":
            self.builder.exp(self._get_var(node)[0])
        else:
            raise RuntimeError("Unrecognized node type '{}'".format(node.name))

    def ast_to_expression(self, node):
        if node.name == "^":
            return "{}^{}".format(self.ast_to_expression(node.children[0]), self.ast_to_expression(node.children[1]))
        elif node.name == "*":
            return "*".join(self.ast_to_xadd(child) for child in node.children)
        elif node.name == "+":
            return "+".join(self.ast_to_xadd(child) for child in node.children)
        elif node.name == "const":
            return self._get_constant(node)
        elif node.name == "var":
            var, v_type = self._get_var(node)
            if v_type not in ["int", "real"]:
                raise RuntimeError("Unsupported variable type '{}' in expression".format(v_type))
            return var
        else:
            raise RuntimeError("Unrecognized node type '{}'".format(node.name))

    @staticmethod
    def _get_constant(node):
        """
        :type node: Node
        """
        v_type = node.children[0].name
        value = node.children[1].name
        if v_type == "bool":
            return bool(value)
        elif v_type == "int":
            return int(value)
        elif v_type == "real":
            return float(value)
        else:
            raise RuntimeError("Unrecognized constant type '{}'".format(v_type))

    def _get_var(self, node):
        var = node.children[1].name
        v_type = node.children[0].name
        # TODO TEMP FIX REMOVE
        if v_type == "real": v_type = "int"
        self.builder.vars(v_type, var)
        return var, v_type
