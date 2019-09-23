"""Gauss-Legendre collocation methods for port-Hamiltonian systems"""

import sympy
import numpy as np

from newton import newton_raphson


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
    c = np.array([float(c_i) for c_i in C])

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
    a = np.array([[float(a_ij) for a_ij in row] for row in A])

    B = [sympy.integrate(l, (tau, 0, 1)) for l in L]
    b = np.array([float(b_j) for b_j in B])

    return a, b, c


def generate_assembly_code(x, F, N, s, a, params):
    """Generates code for the two methods
    compute_residuals and compute_jacobian
    """

    # symbols for Butcher coefficients a_{ij} multiplied by time step h
    asym = [[sympy.Symbol(f'a{i}{j}') for j in range(s)] for i in range(s)]

    # symbols for old state
    osym = [sympy.Symbol(f'o[{n}]') for n in range(N)]

    # symbols for unknowns (flow vector) 
    fsym = [[sympy.Symbol(f'f[{i},{n}]') for n in range(N)] for i in range(s)]

    # polynomial approximation of the numerical solution at the collocation points
    xc = [[(x[n], osym[n] - sum(asym[i][j] * fsym[j][n] for j in range(s))) for n in range(N)] for i in range(s)]

    # expressions for the residuals vector
    residuals = [fsym[i][n] + F[n].subs(xc[i]) for i in range(s) for n in range(N)]

    # expressions for the Jacobian matrix
    jacobian = [[residuals[i].diff(d) for r in fsym for d in r] for i in range(s*N)]

    printer = sympy.printing.lambdarepr.NumPyPrinter()

    code = 'def compute_residuals(residuals, f, o):\n'
    code += f'\tf = f.view()\n\tf.shape = ({s}, {N})\n'
    code += ''.join(f'\ta{i}{j} = {a[i,j]}\n' for i in range(s) for j in range(s))
    code += ''.join(f'\t{symbol} = {value}\n' for symbol, value in params)
    for i in range(s*N):
        code += f'\tresiduals[{i}] = {printer.doprint(residuals[i])}\n' 

    code += '\n\ndef compute_jacobian(jacobian, f, o):\n'
    code += f'\tf = f.view()\n\tf.shape = ({s}, {N})\n'
    code += ''.join(f'\ta{i}{j} = {a[i,j]}\n' for i in range(s) for j in range(s))
    code += ''.join(f'\t{symbol} = {value}\n' for symbol, value in params)
    for i in range(s*N):
        for j in range(s*N):
             code += f'\tjacobian[{i},{j}] = {printer.doprint(jacobian[i][j])}\n'

    return code


def gauss_legendre(x, J, H, x_0, t_f, h, s=1, params=[]):
    """Integrate a port-Hamiltonian system in time
    based on a Gauss-Legendre collocation method. 

    Parameters
    ----------
    x : sympy.Matrix
        vector of symbols for state-space coordinates
    J : sympy.Matrix
        structure matrix
    H : sympy.Expr
        Hamiltonian.
    G : sympy.Matrix
    u : numpy.ndarray
    x_0 : numpy.ndarray
        Initial conditions.
    t_f : float
        Length of time interval.
    h : float
        Desired time step.
    s : int
        Number of stages of the collocation method.
        The resulting method is of order 2s.
    params : List[Tuple[sympy.Symbol, float]]
        Extra parameters on which the system may depend.
    """
    # efforts
    e = sympy.Matrix([H.diff(d) for d in x])

    # dynamics
    F = J @ e

    # number of steps
    K = int(t_f // h)

    # accurate time step
    h = t_f / K

    # dimension of state space
    N = len(x)

    # array for storing time at every step
    time = np.empty(K+1, dtype=float)
    time[0] = t_0 = 0.0

    # array for storing the state at every step
    solution = np.empty((K+1, N), dtype=float)
    solution[0] = x_0

    # flows / unknowns (reused at every step)
    f = np.ones(s*N, dtype=float)
    fmat = f.view()
    fmat.shape = (s, N)

    # residuals vector (reused at every step)
    residuals = np.empty(s*N, dtype=float)

    # jacobian matrix (reused at every step)
    jacobian = np.empty((s*N, s*N), dtype=float)

    # Butcher tableau (multiplied with time step)
    a, b, c = butcher(s)
    a *= h
    b *= h
    c *= h

    # generate methods for assembling the residuals and Jacobian
    ldict = {}
    exec(generate_assembly_code(x, F, N, s, a, params), globals(), ldict)
    compute_residuals = ldict['compute_residuals']
    compute_jacobian = ldict['compute_jacobian']

    for k in range(1, K+1):
        compute_residuals_x = lambda residuals, unknowns : compute_residuals(residuals, unknowns, x_0)
        compute_jacobian_x = lambda jacobian, unknowns : compute_jacobian(jacobian, unknowns, x_0)
        newton_raphson(f, residuals, compute_residuals_x, jacobian, compute_jacobian_x)
        time[k] = t_0 = t_0 + h
        solution[k] = x_0 = x_0 - b @ fmat

    return time, solution
