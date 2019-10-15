"""evaluate a functional defined on the state space"""

import sympy
import numpy
import math

from symbolic import eval_expr


def _generate_code(x, functional):
    """generate code for evaluating functional(x)"""

    printer = sympy.printing.lambdarepr.PythonCodePrinter()

    code = "def evaluate(values, solution):\n"
    code += "\tfor k in range(len(values)):\n"
    code += "\t\t" + ", ".join(str(state) for state in x) + " = solution[k]\n"
    code += f"\t\tvalues[k] = {printer.doprint(functional.evalf())}\n"
    return code


def evaluate_functional(x, functional, solution, functionals={}, params={}):
    """Evaluates a functional on the state space
    for a numerical solution

    Parameters
    ----------
    x : List[sympy.Symbol]
        Symbolic state vector.
    functional : sympy.Expr
        A functional on the state space.
    solution : numpy.ndarray
        Array containing the numerical solution.
        The first axis is time.
        The second axis is state.
    functionals: Dict[sympy.Symbol, sympy.Expr]
        Definitions of functionals.
    params : Dict[sympy.Symbol, Union[sympy.Expr, float]]
        Parameters on which the functional may depend.
    """

    functional = eval_expr(functional, functionals, params)

    undefined = functional.free_symbols - set(x)
    if undefined:
        raise ValueError(
            f"Functional contains symbols with undefined values: {undefined}"
        )

    code = _generate_code(x, functional)
    # print(code)
    ldict = {}
    exec(code, globals(), ldict)
    evaluate = ldict["evaluate"]
    del code, ldict

    K = len(solution)
    values = numpy.empty(K, dtype=float)
    evaluate(values, solution)
    return values
