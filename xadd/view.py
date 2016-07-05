from xadd.diagram import InternalNode, TerminalNode
from xadd.walk import WalkingProfile, ParentsWalker


def to_dot(diagram):
    layers = WalkingProfile.extract_layers(diagram, ParentsWalker(diagram).walk())
    string = "digraph G {\n"
    string += "\trankdir = TB;\n"
    for i in layers:
        for node_id in layers[i]:
            node = diagram.node(node_id)
            if isinstance(node, InternalNode):
                label = node.test
                shape = ""
            elif isinstance(node, TerminalNode):
                label = node.expression
                shape = "box"
            else:
                raise RuntimeError("Unexpected node type: {}".format(type(node)))
            string += "\t{} [label=\"{}\", shape=\"{}\"]\n".format(node_id, label, shape)
            if isinstance(node, InternalNode):
                string += "\t{} -> {}\n".format(node_id, node.child_true)
                string += "\t{} -> {} [style=dashed]\n".format(node_id, node._child_false)
        string += "\t{{rank = same; {}}}\n".format(" ".join(map(lambda n: str(n) + ";", layers[i])))
    string += "}\n"
    return string


def export(diagram, filename):
    with open(filename, "w") as file:
        file.write(to_dot(diagram))
