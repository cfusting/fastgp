import math

from deap import gp
import numpy


def protected_div_one(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def protected_div_zero(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0


def protected_div_dividend(left, right):
    if right != 0:
        return left / right
    else:
        return left


def aq(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, numpy.sqrt(1 + numpy.square(right)))
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = left[numpy.isinf(x)]
            x[numpy.isnan(x)] = left[numpy.isnan(x)]
        elif numpy.isinf(x) or numpy.isnan(x):
            x = left
    return x


def numpy_protected_div_dividend(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = left[numpy.isinf(x)]
            x[numpy.isnan(x)] = left[numpy.isnan(x)]
        elif numpy.isinf(x) or numpy.isnan(x):
            x = left
    return x


def numpy_protected_div_zero(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 0.0
            x[numpy.isnan(x)] = 0.0
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 0.0
    return x


def numpy_protected_div_one(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1.0
            x[numpy.isnan(x)] = 1.0
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1.0
    return x


def numpy_protected_sqrt(x):
            x = numpy.sqrt(x)
            if isinstance(x, numpy.ndarray):
                x[numpy.isnan(x)] = 0
            elif numpy.isnan(x):
                x = 0
            return x


def protected_log_one(x):
    if x > 0:
        return math.log(x)
    else:
        return 1


def protected_log_abs(x):
    if x != 0:
        return math.log(abs(x))
    else:
        return 0


def cube(x):
    return numpy.power(x, 3.0)


def numpy_protected_log_abs(x):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        abs_val = numpy.abs(x)
        x = numpy.log(abs_val.astype(float))
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = -1e300
            x[numpy.isnan(x)] = 0
        elif numpy.isinf(x):
            x = -1e300
        elif numpy.isnan(x):
            x = 0
    return x


def numpy_protected_log_one(x):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.log(numpy.abs(x))
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1.0
            x[numpy.isnan(x)] = 1.0
        elif numpy.isinf(x):
            x = 1.0
        elif numpy.isnan(x):
            x = 1.0
    return x


def get_terminal_order(node, context=None):
    if isinstance(node, gp.Ephemeral) or isinstance(node.value, float) \
            or isinstance(node.value, int) or context is not None and node.value in context:
        return 0
    return 1


def calculate_order(ind, context=None):
    order_stack = []
    for node in reversed(ind):
        if isinstance(node, gp.Terminal):
            terminal_order = get_terminal_order(node, context)
            order_stack.append(terminal_order)
        elif node.arity == 1:
            arg_order = order_stack.pop()
            if node.name == numpy_protected_log_abs.__name__:
                order_stack.append(3 * arg_order)
            elif node.name == numpy.exp.__name__:
                order_stack.append(4 * arg_order)
            else:  # cube or square
                order_stack.append(1.5 * arg_order)
        else:  # node.arity == 2:
            args_order = [order_stack.pop() for _ in range(node.arity)]
            if node.name == numpy.add.__name__ or node.name == numpy.subtract.__name__:
                order_stack.append(max(args_order))
            else:
                order_stack.append(sum(args_order))
    return order_stack.pop()


def get_numpy_infix_symbol_map():
    symbol_map = {numpy.add.__name__: "({0} + {1})",
                  numpy.subtract.__name__: "({0} - {1})",
                  numpy.multiply.__name__: "({0} * {1})",
                  numpy_protected_div_dividend.__name__: "({0} / {1})",
                  numpy_protected_log_abs.__name__: "log({0})",
                  numpy.abs.__name__: "abs({0})",
                  numpy.sin.__name__: "sin({0})",
                  numpy.cos.__name__: "cos({0})",
                  numpy.exp.__name__: "exp({0})",
                  numpy.square.__name__: "(({0}) ^ 2)",
                  cube.__name__: "(({0}) ^ 3)",
                  numpy.sqrt.__name__: "sqrt({0})",
                  numpy.reciprocal.__name__: "(1 / {0})",
                  aq.__name__: "({0} // {1})",
                  numpy.power.__name__: "(({0}) ^ {1})"}
    return symbol_map


def get_numpy_prefix_symbol_map():
    symbol_map = [("+", numpy.add.__name__,),
                  ("-", numpy.subtract.__name__),
                  ("**", numpy.power.__name__),
                  ("^", numpy.power.__name__),
                  ("*", numpy.multiply.__name__),
                  ("/", numpy_protected_div_dividend.__name__),
                  ('abs', numpy.abs.__name__),
                  ("log", numpy_protected_log_abs.__name__),
                  ("sin", numpy.sin.__name__),
                  ("cos", numpy.cos.__name__),
                  ("exp", numpy.exp.__name__)]
    return symbol_map


def get_numpy_commutative_set():
    return {numpy.add.__name__, numpy.multiply.__name__}
