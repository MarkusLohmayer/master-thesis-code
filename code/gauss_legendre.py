"""Gauss-Legendre collocation methods for port-Hamiltonian systems"""

import sympy
import numpy
import math

from newton import newton_raphson, DidNotConvergeError
from symbolic import eval_expr


def butcher(s):
    """Compute the Butcher tableau for a Gauss-Legendre collocation method.

    Parameters
    ----------
    s : int
        Number of stages of the collocation method.
        The resulting method is of order 2s.

    Returns
    -------
    a : numpy.ndarray
        Coefficients a_{ij}, i.e. the j-th lagrange polynomial integrated on (0, c_i).
    b : numpy.ndarray
        Coefficients b_j, i.e. the the i-th lagrange polynomial integrated on (0, 1).
    c : numpy.ndarray
        Coefficients c_i, i.e. the collocation points.
    """

    from sympy.abc import tau, x

    # shifted Legendre polynomial of order s
    P = (x ** s * (x - 1) ** s).diff(x, s)

    # roots of P
    C = sympy.solve(P)
    C.sort()
    c = numpy.array([float(c_i) for c_i in C])

    # Lagrange basis polynomials at nodes C
    L = []
    for i in range(s):
        l = 1
        for j in range(s):
            if j != i:
                l = (l * (tau - C[j]) / (C[i] - C[j])).simplify()
        L.append(l)

    # integrals of Lagrange polynomials
    A = [[sympy.integrate(l, (tau, 0, c_i)) for l in L] for c_i in C]
    a = numpy.array([[float(a_ij) for a_ij in row] for row in A])

    B = [sympy.integrate(l, (tau, 0, 1)) for l in L]
    b = numpy.array([float(b_j) for b_j in B])

    return a, b, c


def gauss_legendre(x, F, x_0, t_f, dt, s=1, functionals={}, params={}):
    """Integrate a port-Hamiltonian system in time
    based on a Gauss-Legendre collocation method.

    Parameters
    ----------
    x : sympy.Matrix
        vector of symbols for state-space coordinates
    F : List[sympy.Expr]
        The right-hand side of the ODE.
    x_0 : numpy.ndarray
        Initial conditions.
    t_f : float
        Length of time interval.
    dt : float
        Desired time step.
    s : int
        Number of stages of the collocation method.
        The resulting method is of order 2s.
    functionals : Dict[sympy.Symbol, sympy.Expr]
        Functionals on which F may depend.
    params : Dict[sympy.Symbol, Union[sympy.Expr, float]]
        Parameters on which the system may depend.
    """

    # number of steps
    K = int(t_f // dt)

    # accurate time step
    dt = t_f / K

    # dimension of state space
    N = len(x)

    # Butcher tableau (multiplied with time step)
    a, b, c = butcher(s)
    a *= dt
    b *= dt
    c *= dt

    # generate code for evaluating residuals vector and Jacobian matrix
    code = _generate_code(x, F, N, a, s, functionals, params)
    # print(code)
    # return None, None
    ldict = {}
    exec(code, None, ldict)
    compute_residuals = ldict["compute_residuals"]
    compute_jacobian = ldict["compute_jacobian"]
    del code, ldict

    # array for storing time at every step
    time = numpy.empty(K + 1, dtype=float)
    time[0] = t_0 = 0.0

    # array for storing the state at every step
    solution = numpy.empty((K + 1, N), dtype=float)
    solution[0] = x_0

    # flows / unknowns (reused at every step)
    f = numpy.ones(s * N, dtype=float)
    fmat = f.view()
    fmat.shape = (s, N)

    # residuals vector (reused at every step)
    residuals = numpy.empty(s * N, dtype=float)

    # jacobian matrix (reused at every step)
    jacobian = numpy.empty((s * N, s * N), dtype=float)

    for k in range(1, K + 1):
        try:
            newton_raphson(
                f,
                residuals,
                lambda residuals, unknowns: compute_residuals(residuals, unknowns, x_0),
                jacobian,
                lambda jacobian, unknowns: compute_jacobian(jacobian, unknowns, x_0),
                tol=1e-9,
                iterations=500,
            )
        except DidNotConvergeError:
            print(f"Did not converge at step {k}.")
            break
        time[k] = t_0 = t_0 + dt
        solution[k] = x_0 = x_0 - b @ fmat

    return time, solution


def _generate_code(x, F, N, a, s, functionals, params):
    """Generate code for the two methods compute_residuals and compute_jacobian"""

    # dynamics
    F = [eval_expr(f, functionals) for f in F]

    # symbols for Butcher coefficients a_{ij} multiplied by time step h
    asym = [[sympy.Symbol(f"a{i}{j}") for j in range(s)] for i in range(s)]

    # symbols for old state
    osym = [sympy.Symbol(f"o[{n}]") for n in range(N)]

    # symbols for unknowns (flow vector)
    fsym = [[sympy.Symbol(f"f[{i},{n}]") for n in range(N)] for i in range(s)]

    # polynomial approximation of the numerical solution at the collocation points
    xc = [
        [
            (x[n], osym[n] - sum(asym[i][j] * fsym[j][n] for j in range(s)))
            for n in range(N)
        ]
        for i in range(s)
    ]

    # expressions for the residuals vector
    residuals = [fsym[i][n] + F[n].subs(xc[i]) for i in range(s) for n in range(N)]

    # expressions for the Jacobian matrix
    jacobian = [[residuals[i].diff(d) for r in fsym for d in r] for i in range(s * N)]

    printer = sympy.printing.lambdarepr.PythonCodePrinter()

    code = "def compute_residuals(residuals, f, o):\n"
    code += f"\tf = f.view()\n\tf.shape = ({s}, {N})\n"
    code += "".join(f"\ta{i}{j} = {a[i,j]}\n" for i in range(s) for j in range(s))
    # code += "".join(f"\t{symbol} = {printer.doprint(value)}\n" for symbol, value in params.items())
    for i in range(s * N):
        code += f"\tresiduals[{i}] = {printer.doprint(eval_expr(residuals[i], params=params).evalf())}\n"
        # code += f"\tresiduals[{i}] = {printer.doprint(residuals[i])}\n"

    code += "\n\ndef compute_jacobian(jacobian, f, o):\n"
    code += f"\tf = f.view()\n\tf.shape = ({s}, {N})\n"
    code += "".join(f"\ta{i}{j} = {a[i,j]}\n" for i in range(s) for j in range(s))
    # code += "".join(f"\t{symbol} = {printer.doprint(value)}\n" for symbol, value in params.items())
    for i in range(s * N):
        for j in range(s * N):
            code += f"\tjacobian[{i},{j}] = {printer.doprint(eval_expr(jacobian[i][j], params=params).evalf())}\n"
            # code += f"\tjacobian[{i},{j}] = {printer.doprint(jacobian[i][j])}\n"

    return code
