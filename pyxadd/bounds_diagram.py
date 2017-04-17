from pyxadd import build
from pyxadd import diagram
from pyxadd import view
from pyxadd import test
from pyxadd import order
from pyxadd import operation
from pyxadd import leaf_transform
from pyxadd import reduce

import sympy
ub_cache = {}
lb_cache = {}
pool = diagram.Pool()

def two_var_diagram():
    bounds = b.test("x", ">=", 0) & b.test("x", "<=", 1)
    bounds &= b.test("y", ">=", 1) & b.test("y", "<=", 3)
    two = b.test("x", ">=", "y")
    return b.ite(bounds, b.ite(two, b.terminal("x"), b.terminal("10")), b.terminal(0))

b = build.Builder(pool)
b.ints("x", "y", "a", "b", "c", "ub", "lb", "bla", "ub", "lb")
diagram1 = b.ite(b.test("x", "<=", "a"),
                b.ite(b.test("x", ">=", "b"),
                      b.exp("ub - lb"), b.exp(0)),
                b.ite(b.test("x", "<=", "c"),
                      b.exp("(ub - lb)**2"),  b.exp(0))
                )
diagram2 = b.ite(b.test("x", ">=", "b"), b.exp("ub - lb"), b.exp(0))
bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
# d = b.ite(bounds, b.terminal("x"), b.terminal(0))

d = two_var_diagram()

def recurse(node_id):
  node = pool.get_node(node_id)
  if node.is_terminal():
    print(node.expression)
  else:
    print(node.test)
    if isinstance(node.test, test.LinearTest):
       print("Coefficients of x ({}) and y ({})".format(node.test.operator.coefficient("x"), node.test.operator.coefficient("y")))
    print "t", recurse(node.child_true)
    print "f", recurse(node.child_false)

def integrate(node_id, var):
  def symbolic_integrator(terminal_node, d):
    assert isinstance(terminal_node, diagram.TerminalNode)
    return d.pool.terminal(sympy.Sum(terminal_node.expression, (var, "lb", "ub")).doit())
  return resolve_lb_ub(leaf_transform.transform_leaves(symbolic_integrator, pool.diagram(node_id)), var)


def resolve_lb_ub(node_id, var):
  
  node = pool.get_node(node_id)
  # leaf
  if node.is_terminal():
    return node.node_id
  # not leaf
  var_coefficient = node.test.operator.coefficient(var)
  if var_coefficient != 0:
    if  var_coefficient > 0:
      # split into cases
      # case 1: root.test is smaller than all other UBs
      # -> we make root larger than all other ubs, and we determine the lb
      best_ub = resolve_lb(dag_resolve(var, node.test.operator,
                                       node.child_true, "leq", "ub"),
                           var, node.test.operator)
      # case 2: root.test is not going to be the ub, we need to make root
      # -> it needs to be lower than all other ubs; bound_max will append
      # a comparison of node.test.operator > ub(leaf) before each leaf.
      some_ub = bound_max(node.test.operator,
                          resolve_lb_ub(node.child_true, var), var)
      # case 3: test is false, then root.test is the lower bound
      # -> make root.test larger than all other lbs
      best_lb = resolve_ub(dag_resolve(var, (~node.test.operator).to_canonical(),
                                       node.child_false, "geq", "lb"),
                           var, (~node.test.operator).to_canonical())
      # case 4:
      some_lb = bound_min((~node.test.operator).to_canonical(),
                          resolve_lb_ub(node.child_false, var), 
                          var)
    else:
      # split into cases
      # case 1:
      best_lb = resolve_ub(dag_resolve(var, node.test.operator,
                                       node.child_true, "geq", "lb"),
                           var, node.test.operator)
      # case 2: 
      some_lb = bound_min(node.test.operator,
                          resolve_lb_ub(node.child_true, var),
                          var)
      #view.export(pool.diagram(some_lb), "../../Dropbox/XADD Matrices/debug.dot".format(str(node.test.operator)))
      #print "Asdf"
      #exit()
      # case 3: 
      best_ub = resolve_lb(dag_resolve(var, (~node.test.operator).to_canonical(),
                                       node.child_false, "leq", "ub"),
                           var, (~node.test.operator).to_canonical())
      # case 4:
      some_ub = bound_max((~node.test.operator).to_canonical(), 
                          resolve_lb_ub(node.child_false, var),
                          var)
    return (pool.diagram(some_lb) 
           + pool.diagram(best_lb)
           + pool.diagram(some_ub) 
           + pool.diagram(best_ub)).root_id
  else:
    print "indep brach ublb", node.test.operator
    test_node_id = pool.bool_test(node.test)
    return pool.apply(operation.Summation,
      pool.apply(operation.Multiplication, test_node_id, resolve_lb_ub(node.child_true, var)),
      pool.apply(operation.Multiplication, pool.invert(test_node_id), resolve_lb_ub(node.child_false, var)))


#view.export(pool.diagram(some_ub), "../../Dropbox/XADD Matrices/dr_1_someub_{}.dot".format(str(node.test.operator)))
def resolve_ub(node_id, var, lower_bound, noprint=True):
  node = pool.get_node(node_id)
  if node.is_terminal():
    return node_id 
  var_coefficient = node.test.operator.coefficient(var)
  if var_coefficient != 0:
    if var_coefficient > 0:
      best_ub = dag_resolve(var, node.test.operator, node.child_true, "leq", "ub", consume=True)
      some_ub = bound_max(node.test.operator,
                          resolve_ub(node.child_true, var, lower_bound), var)
      non_ub = resolve_ub(node.child_false, var, lower_bound)
      print "res", lower_bound, node.test.operator
      resolve_test = resolve(var, lower_bound, node.test.operator, "", "ub")
      print("Resolve test (resolve_ub true branch) {}".format(pool.get_node(resolve_test)))
      res = b.ite(pool.diagram(resolve_test), 
                  pool.diagram(best_ub) + pool.diagram(some_ub),
                  pool.diagram(non_ub)
                  )
    else:
      best_ub = dag_resolve(var, (~node.test.operator).to_canonical(),
                            node.child_false, "leq", "ub", consume=True)
      some_ub = bound_max((~node.test.operator).to_canonical(),
                          resolve_ub(node.child_false, var, lower_bound), var)
      non_ub = resolve_ub(node.child_true, var, lower_bound)
      resolve_test = resolve(var, lower_bound, (~node.test.operator).to_canonical(), "", "ub") 
      print("Resolve test (resolve_ub else branch) {}".format(pool.get_node(resolve_test)))
      res = b.ite(pool.diagram(resolve_test),
                  pool.diagram(non_ub),
                  pool.diagram(best_ub) + pool.diagram(some_ub)
                  )
    return res.root_id
  else:
    print "indep brach ub", node.test.operator
    test_node_id = pool.bool_test(node.test)
    return pool.apply(operation.Summation,
      pool.apply(operation.Multiplication, test_node_id, resolve_ub(node.child_true, var, lower_bound)),
      pool.apply(operation.Multiplication, pool.invert(test_node_id), resolve_ub(node.child_false, var, lower_bound)))

def resolve_lb(node_id, var, upper_bound):
  node = pool.get_node(node_id)
  if node.is_terminal():
    return pool.zero_id
  var_coefficient = node.test.operator.coefficient(var)
  if var_coefficient != 0:
    if var_coefficient < 0:
      best_lb = dag_resolve(var, node.test.operator, node.child_true, "geq", "lb", consume=True)
      some_lb = bound_min(node.test.operator,
                          resolve_lb(node.child_true, var, upper_bound), var)
      non_lb = resolve_lb(node.child_false, var, upper_bound)
      resolve_test = resolve(var, upper_bound, node.test.operator, "", "lb")
      print("Resolve test (resolve_lb true branch) {}".format(pool.get_node(resolve_test)))
      res = b.ite(pool.diagram(resolve_test),
                  pool.diagram(best_lb) + pool.diagram(some_lb),
                  pool.diagram(non_lb)
                  )
    else:
      best_lb = dag_resolve(var, (~node.test.operator).to_canonical(), node.child_false, "geq", "lb")
      some_lb = bound_min((~node.test.operator).to_canonical(),
                          resolve_lb(node.child_false, var, upper_bound), var)
      non_lb = resolve_lb(node.child_true, var, upper_bound)
      resolve_test = resolve(var, upper_bound, (~node.test.operator).to_canonical(), "", "lb")
      print("Resolve test (resolve_lb else branch) {}".format(pool.get_node(resolve_test)))
      res = b.ite(pool.diagram(resolve_test), 
                  pool.diagram(non_lb),
                  pool.diagram(best_lb) + pool.diagram(some_lb)
                  )
    return res.root_id
  else:
    print "indep brach lb", node.test.operator
    test_node_id = pool.bool_test(node.test)
    return pool.apply(operation.Summation,
      pool.apply(operation.Multiplication, test_node_id, resolve_lb(node.child_true, var, upper_bound)),
      pool.apply(operation.Multiplication, pool.invert(test_node_id), resolve_lb(node.child_false, var, upper_bound)))

def to_exp(op, var):
  expression = sympy.sympify(op.rhs)
  for k, v in op.lhs.items():
    if k != var:
      expression = -sympy.S(k) * v + expression
  return expression

def operator_to_bound(operator, var):
  # TODO Check for integer division
  bound = operator.times(1 / operator.coefficient(var)).weak()
  exp_pos = to_exp(bound, var) 
  return exp_pos

def simplify(linear_test, true_diagram, false_diagram):
    if linear_test.operator.is_tautology():
        if linear_test.operator.rhs < 0:
            # Infeasible
            return false_diagram
        else:
            # Tautoogy
            return true_diagram
    else:
        return b.ite(pool.diagram(pool.bool_test(linear_test)), true_diagram, false_diagram)

def bound_min(operator, node_id, var):
  node = pool.get_node(node_id)
  bound = operator_to_bound(operator, var)
  def leq_leaf(lb_cache, bound, node, diagram):
    if not node.is_terminal():
      raise RuntimeError("Node not terminal, wtf")
    if node.node_id in lb_cache:
      res = simplify(test.LinearTest(lb_cache[node.node_id], ">", bound),
                  b.exp(node.expression), b.exp(sympy.sympify("0")))
      print("Created bound (min) {}".format(res.root_node))
      return res.root_id
    else: return pool.zero_id
  if node.is_terminal():
    return leq_leaf(lb_cache, bound, node, pool.diagram(node_id))
  else: 
    return leaf_transform.transform_leaves(lambda x, y: leq_leaf(lb_cache, bound, x, y), pool.diagram(node_id))

def bound_max(operator, node_id, var):
  node = pool.get_node(node_id)
  bound = operator_to_bound(operator, var)
  def geq_leaf(ub_cache, bound, node, diagram):
    if not node.is_terminal():
      raise RuntimeError("Node not terminal, wtf")
    if node.node_id in ub_cache:
      res = simplify(test.LinearTest(ub_cache[node.node_id], "<", bound),
                  b.exp(node.expression), b.exp(sympy.sympify("0")))
      print("Created bound (max) {}".format(res.root_node))
      return res.root_id
    else: return pool.zero_id
  if node.is_terminal():
    return geq_leaf(ub_cache, bound, node, pool.diagram(node_id))
  else: 
    return leaf_transform.transform_leaves(lambda x, y: geq_leaf(ub_cache, bound, x, y), pool.diagram(node_id))


def dag_resolve(var, operator, node_id, direction, bound_type, substitute=False, consume=False):
  node = pool.get_node(node_id)
  if node.is_terminal():
    res = pool.terminal(node.expression.subs({bound_type: operator_to_bound(operator, var)}))
    if bound_type == "ub":
      ub_cache[res] = operator
    else:
      lb_cache[res] = operator
    return res
  var_coefficient = node.test.operator.coefficient(var)
  if var_coefficient != 0:
    test_node_id = pool.bool_test(node.test)
    resolve_true = resolve(var, operator, node.test.operator, direction, bound_type)
    resolve_false = resolve(var, operator, (~node.test.operator).to_canonical(), direction, bound_type) 
    dr_true = dag_resolve(var, operator, node.child_true, direction,
                          bound_type, substitute=substitute, consume=consume)
    dr_false = dag_resolve(var, operator, node.child_false, direction,
                           bound_type, substitute=substitute, consume=consume)
    if consume:
      test_diagram = pool.diagram(resolve_true) * pool.diagram(resolve_false)
      res = b.ite(test_diagram,
                  pool.diagram(dr_true),
                  pool.diagram(dr_false)
                  )
    else:
      res = b.ite(pool.diagram(test_node_id),
                  pool.diagram(resolve_true) * pool.diagram(dr_true),
                  pool.diagram(resolve_false) * pool.diagram(dr_false)
                  )
    return res.root_id
  else:
    test_node_id = pool.bool_test(node.test)
    return pool.apply(operation.Summation,
      pool.apply(operation.Multiplication, test_node_id, dag_resolve(var, operator, node.child_true, direction, bound_type, consume=consume, substitute=substitute)),
      pool.apply(operation.Multiplication, pool.invert(test_node_id), dag_resolve(var, operator, node.child_false, direction, bound_type, consume=consume, substitute=substitute)))

def resolve(var, operator_rhs, operator_lhs, direction, bound_type):
  lhs_coefficient = operator_lhs.coefficient(var)
  lhs_type = "na"
  if lhs_coefficient > 0:
    lhs_type = "ub"
  elif lhs_coefficient < 0:
    lhs_type = "lb"
  else:
    raise RuntimeError("Variable {} does not appear in expression {}".format(var, operator_lhs))
  if lhs_type != bound_type:
    return pool.terminal(1)
  else:
    if direction == "geq":
      res = operator_rhs.resolve(var, operator_lhs.switch_direction())
    elif direction == "leq":
      res = operator_lhs.resolve(var, operator_rhs.switch_direction())
    else:
      res = operator_lhs.resolve(var, operator_rhs)
    res = res.to_canonical()
    # print ",".join([str(u) for u in [operator_rhs, operator_lhs, res]])
    print("Resolving ({}) and ({}) = ({})".format(operator_rhs, operator_lhs, res))
    if res.is_tautology():
        if res.rhs < 0:
            return pool.terminal(0)
        else:
            return pool.terminal(1)
    # zero = True
    # for var in res.variables:
    #   if res.coefficient != 0:
    #     zero = False
    #     break
    # if zero:
    #   return pool.terminal(1)
    return pool.bool_test(test.LinearTest(res))




operator_1 = test.LinearTest("x", "<=", "a").operator
operator_2 = test.LinearTest("x", "<=", "2").operator

resolved_node_id = resolve("x", operator_1, operator_2, "leq", "ub")
print(pool.get_node(resolved_node_id).test.operator)
  
resolved_node_id = resolve("x", operator_1, operator_2, "geq", "ub")

print(pool.get_node(resolved_node_id).test.operator)

#dr = dag_resolve("x", operator_1, pool.bool_test(test.LinearTest(operator_2)), "geq", "ub")
#print("Diagram is {}ordered".format("" if order.is_ordered(pool.diagram(dr)) else "not "))
#view.export(pool.diagram(dr), "../../Dropbox/XADD Matrices/test.dot")
test_diagram = d
view.export(test_diagram, "diagram.dot")
#dr = dag_resolve("x", operator_1, diagram.root_id, "leq", "ub")
#view.export(pool.diagram(dr), "../../Dropbox/XADD Matrices/dr.dot")
# recurse(diagram.root_id)
dr = integrate(test_diagram.root_id, "x")
view.export(pool.diagram(dr), "result.dot")

dr = reduce.LinearReduction(pool).reduce(dr)
#fm = fourier_motzkin(bounds.root_id, "ub")
view.export(pool.diagram(dr), "result_reduced.dot")

d_const = pool.diagram(dr)
for y in range(-20, 20):
    s = 0
    for x in range(-20, 20):
        s += d.evaluate({"x": x, "y": y})
    print(s - d_const.evaluate({"y": y}))
