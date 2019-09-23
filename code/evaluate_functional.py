"""evaluate a functional defined on the state space"""

import sympy
import numpy as np


def evaluate_functional(x, functional, solution, params=[]):
    """Evaluates a functional on the state space
    for a numerical solution

    Parameters
    ----------
    x : List[sympy.Symbol]
        State-space coordinates.
    functional : sympy.Expr
        A functional on the state space.
    solution : numpy.ndarray
        Array containing the numerical solution.
        The first axis is time.
        The second axis is state.
    params : List[Tuple[sympy.Symbol, float]]
        Extra parameters on which the functional may depend.
    """

    free_symbols = functional.free_symbols - set(x) - set(s for s, _ in params)
    if free_symbols:
        raise Exception(f"functional contains free symbols: {free_symbols}")

    printer = sympy.printing.lambdarepr.NumPyPrinter()

    code = "def evaluate(x):\n"
    if params:
        code += "".join(f"\t{symbol} = {value}\n" for symbol, value in params)
    code += "".join(f"\t{symbol} = x[{i}]\n" for i, symbol in enumerate(x))
    code += f"\treturn {printer.doprint(functional)}"

    ldict = {}
    exec(code, globals(), ldict)
    evaluate = ldict["evaluate"]

    K = len(solution)
    values = np.empty(K, dtype=float)
    for k in range(K):
        values[k] = evaluate(solution[k])

    return values
