import re

from pyxadd.test import Test, LinearTest


class Node(object):
    def __init__(self, name=None):
        self.name = name
        self.children = []

    def has_name(self):
        return self.name is not None

    def __repr__(self):
        return "Node({}, {})".format(self.name, self.children)


def tokenize(chars):
    """Convert a string of characters into a list of tokens."""
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()


def file_to_tokens(filename):
    trees = []
    with open(filename) as stream:
        for line in stream:
            if line is not "":
                tree_string = line
                tree_string = re.sub(r"\(\(", "((feature, ", tree_string)
                tree_string = re.sub(r"(tree|class)\(", r"(\g<1>, ", tree_string)
                tree_string = re.sub(r"\s+", "", tree_string)
                tree_string = re.sub(r",", " ", tree_string)
                trees.append(tokenize(tree_string))
    return trees


def tokenized_string_to_ast(tokenized_string):
    stack = []
    root = None
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
                current.name = token
    return root


def ast_to_eval(root_node):
    assert isinstance(root_node, Node)
    if root_node.name == "tree":
        # Recur
        test, child_true, child_false = map(ast_to_eval, root_node.children)
        return lambda assignment: child_true(assignment) if test(assignment) else child_false(assignment)
    elif root_node.name == "feature":
        # Construct test
        name = root_node.children[0].name
        value = int(float(root_node.children[1].name))
        return lambda assignment: assignment[name] <= value
    elif root_node.name == "class":
        # Construct result
        return lambda assignment: int(root_node.children[0].name)


def ast_to_xadd(pool, root_node):
    # [(sum_of_papers1, 0, 400), (median_year1, 1950, 2016), (sum_of_papers2, 0, 400), (median_year2, 1950, 2016)]
    """
    Converts the abstract syntax tree to an xadd
    :type root_node: Node
    :type pool: pyxadd.diagram.Pool
    """
    assert isinstance(root_node, Node)
    if root_node.name == "tree":
        # Recur
        test, child_true, child_false = map(lambda root_node: ast_to_xadd(pool, root_node), root_node.children)
        return pool.internal(test, child_true, child_false)
    elif root_node.name == "feature":
        # Construct test
        name = root_node.children[0].name
        value = int(float(root_node.children[1].name))
        return LinearTest(name, "<=", value)
    elif root_node.name == "class":
        # Construct result
        return pool.terminal(int(root_node.children[0].name))


def read_trees(filename):
    tokenized_strings = file_to_tokens(filename)
    abstract_syntax_trees = map(tokenized_string_to_ast, tokenized_strings)
    decision_trees = map(ast_to_eval, abstract_syntax_trees)
    return decision_trees


def read_xadds(filename, pool):
    from pyxadd.order import order

    tokenized_strings = file_to_tokens(filename)
    abstract_syntax_trees = map(tokenized_string_to_ast, tokenized_strings)
    root_node_ids = map(lambda root_node: ast_to_xadd(pool, root_node), abstract_syntax_trees)
    decision_trees = list(order(pool.diagram(node_id)) for node_id in root_node_ids)
    return decision_trees

