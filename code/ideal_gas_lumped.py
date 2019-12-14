"""ideal gas model for lumped volumes
based on the Sackur-Tetrode equation

fundamental equation: u = U(s, v, m)
"""

import sympy


# natural constants
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

# atomic weight
_m_a = sympy.Symbol("m_a", real=True, positive=True)

# state variables
_s = sympy.Symbol("s", real=True, positive=True)
_v = sympy.Symbol("v", real=True, positive=True)
_m = sympy.Symbol("m", real=True, positive=True)

# fundamental equation
_U = sympy.Symbol("U", real=True, positive=True)

# equations of state
_θ = sympy.Symbol("θ", real=True, positive=True)
_π = sympy.Symbol("π", real=True, positive=True)
_μ = sympy.Symbol("μ", real=True, positive=True)

_functionals = {
    # internal energy U(s, v, m)
    _U: 3
    / (4 * sympy.pi)
    * sympy.exp(-sympy.Rational(5, 3))
    * h ** 2
    * _m_a ** -sympy.Rational(8, 3)
    * _m ** sympy.Rational(5, 3)
    / _v ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # temperature θ(s, v, m)
    _θ: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2 / k_B
    * _m_a ** -sympy.Rational(5, 3)
    * (_m / _v) ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # pressure π(s, v, m)
    _π: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2
    * _m_a ** -sympy.Rational(8, 3)
    * (_m / _v) ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # chemical potential μ(s, v, m)
    _μ: sympy.exp(-sympy.Rational(5, 3))
    / (4 * sympy.pi)
    * h ** 2 / k_B
    * _m_a ** -sympy.Rational(8, 3)
    * (5 * k_B * _m - 2 * _m_a * _s)
    * (_m * _v**2) ** -sympy.Rational(1, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
}


def add_functionals(functionals, U, s, v, m, m_a, θ=None, π=None, μ=None):
    """add functionals for ideal gas in a lumped volume
    to a dictionary of functionals.
    The functional U(s, v, m) will always be added.
    The functionals θ(s, v, m), π(s, v, m) and μ(s, v, m)
    will only be added if a symbol is provided.

    Parameters
    ----------
    functionals : dict
        A dictionary to which the functionals are added.
    U : sympy.Symbol
        Symbol for internal energy of the gas (J).
    s : sympy.Symbol
        Symbol for entropy of the gas (J/K).
    v : sympy.Symbol
        Symbol for volume of the gas (m**3).
    m : sympy.Symbol
        Symbol for mass of the gas (kg).
    m_a : sympy.Symbol
        Symbol for atomic mass (kg).
    θ : sympy.Symbol or None
        Symbol for temerature (K).
    π : sympy.Symbol or None
        Symbol for pressure (Pa).
    μ : sympy.Symbol or None
        Symbol for chemical potential per unit mass (J/kg).
    """

    subs = {
        _s: s,
        _v: v,
        _m: m,
        _m_a: m_a,
    }

    functionals[U] = _functionals[_U].subs(subs)

    if θ:
        functionals[θ] = _functionals[_θ].subs(subs)
    if π:
        functionals[π] = _functionals[_π].subs(subs)
    if μ:
        functionals[μ] = _functionals[_μ].subs(subs)
