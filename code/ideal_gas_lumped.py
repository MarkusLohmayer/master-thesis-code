"""ideal gas model for lumped volumes
based on the Sackur-Tetrode equation

fundamental equation: U = U(S, V, M)
"""

import sympy


# state-independent constants
h = sympy.Symbol("h", real=True, positive=True)
k_B = sympy.Symbol("k_B", real=True, positive=True)
N_A = sympy.Symbol("N_A", real=True, positive=True)

# atmomic weights (from Periodic table)
m_He = sympy.Symbol("m_He", real=True, positive=True)
m_Ar = sympy.Symbol("m_Ar", real=True, positive=True)

params = {
    # Planck constant (Js)
    h: 6.62607015e-34,
    # Boltzmann constant (J/K)
    k_B: 1.380649e-23,
    # Avogadro constant (1/mol)
    N_A: 6.02214076 * 10 ** 23,
    # weight of one Helium atom (kg)
    m_He: 4.00 / (1000 * N_A),
    # weight of one Argon atom (kg)
    m_Ar: 39.95 / (1000 * N_A),
}

# general atomic weight
_m = sympy.Symbol("m_atom", real=True, positive=True)

# state variables
_S = sympy.Symbol("S", real=True, positive=True)
_V = sympy.Symbol("V", real=True, positive=True)
_M = sympy.Symbol("M", real=True, positive=True)

# fundamental equation
_U = sympy.Symbol("U", real=True, positive=True)

# equations of state
_T = sympy.Symbol("T", real=True, positive=True)
_p = sympy.Symbol("p", real=True, positive=True)
_mu = sympy.Symbol("mu", real=True, positive=True)

_functionals = {
    # internal energy U(S, V, M)
    _U: 3
    / (4 * sympy.pi)
    * sympy.exp(-sympy.Rational(5, 3))
    * h ** 2
    * _m ** -sympy.Rational(8, 3)
    * _M ** sympy.Rational(5, 3)
    / _V ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m / k_B * _S / _M),
    # temperature T(S, V, M)
    _T: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2 / k_B
    * _m ** -sympy.Rational(5, 3)
    * (_M / _V) ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m / k_B * _S / _M),
    # pressure p(S, V, M)
    _p: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2
    * _m ** -sympy.Rational(8, 3)
    * (_M / _V) ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m / k_B * _S / _M),
    # chemical potential mu(S, V, M)
    _mu: sympy.exp(-sympy.Rational(5, 3))
    / (4 * sympy.pi)
    * h ** 2 / k_B
    * _m ** -sympy.Rational(8, 3)
    * (5 * k_B * _M - 2 * _m * _S)
    * (_M * _V**2) ** -sympy.Rational(1, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m / k_B * _S / _M),
}


def add_functionals(functionals, U, S, V, M, m, T=None, p=None, mu=None):
    """add functionals for ideal gas in a lumped volume
    to a dictionary of functionals.
    The functional U(S, V, M) will always be added.
    The functionals T(S, V, M), p(S, V, M) and mu(S, V, M)
    will only be added if a symbol is provided.

    Parameters
    ----------
    functionals : dict
        A dictionary to which the functionals are added.
    U : sympy.Symbol
        Symbol for internal energy of the gas (J).
    S : sympy.Symbol
        Symbol for ntropy of the gas (J/K).
    V : sympy.Symbol
        Symbol for volume of the gas (m**3).
    M : sympy.Symbol
        Symbol for mass of the gas (kg).
    m : sympy.Symbol
        Symbol for atomic mass (kg).
    T : sympy.Symbol or None
        Symbol for temerature (K).
    p : sympy.Symbol or None
        Symbol for pressure (Pa).
    mu : sympy.Symbol or None
        Symbol for chemical potential per unit mass (J/kg).
    """

    subs = {
        _S: S,
        _V: V,
        _M: M,
        _m: m,
    }

    functionals[U] = _functionals[_U].subs(subs)

    if T:
        functionals[T] = _functionals[_T].subs(subs)
    if p:
        functionals[p] = _functionals[_p].subs(subs)
    if mu:
        functionals[mu] = _functionals[_mu].subs(subs)
