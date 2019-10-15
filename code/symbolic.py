"""helper functions for symbolic manipulation of equations"""

import sympy


def eval_expr(expr, functionals={}, params={}, state={}):
    """Evaluate a symbolic expression based provided information.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to evaluate.
    functionals: Dict[sympy.Symbol, sympy.Expr]
        Dictionary of known functionals.
    params: Dict[sympy.Symbol, Union[sympy.Expr, float]]
        Dictionary of known parameters.
    state : Dict[symbol.Symbol, float] or Tuple[List[sympy.Symbol], array_like]
        Optionally provide (part of) the state as a dictionary
        or as a tuple of the symbolic state vector and an array of corresponding values.

    Returns the evaluated expression.
    """

    if state and isinstance(state, tuple):
        state_symbols, state_values = state
        state = {symbol: value for symbol, value in zip(state_symbols, state_values)}

    def replace_unknowns(expression):
        if isinstance(expression, sympy.Expr):
            for symbol in expression.free_symbols:
                replacement = None

                if functionals and symbol in functionals.keys():
                    replacement = functionals[symbol]
                elif params and symbol in params.keys():
                    replacement = params[symbol]
                elif state and symbol in state.keys():
                    replacement = state[symbol]

                if replacement is not None:
                    replacement = replace_unknowns(replacement)
                    expression = expression.subs(symbol, replacement)
        return expression

    return replace_unknowns(expr)
