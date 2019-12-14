"""ideal gas model for distributed systems
based on the Sackur-Tetrode equation

fundamental equation: u = U(s, m)
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
_m = sympy.Symbol("m", real=True, positive=True)

# fundamental equation
_U = sympy.Symbol("U", real=True, positive=True)

# equations of state
_θ = sympy.Symbol("θ", real=True, positive=True)
_π = sympy.Symbol("π", real=True, positive=True)
_μ = sympy.Symbol("μ", real=True, positive=True)

functionals = {
    # internal energy U(s, m)
    _U: 3
    / (4 * sympy.pi)
    * sympy.exp(-sympy.Rational(5, 3))
    * h ** 2
    * _m_a ** -sympy.Rational(8, 3)
    * _m ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # temperature θ(s, m)
    _θ: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2 / k_B
    * _m_a ** -sympy.Rational(5, 3)
    * _m ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # pressure π(s, m)
    _π: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2
    * _m_a ** -sympy.Rational(8, 3)
    * _m ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
    # chemical potential μ(s, m)
    _μ: sympy.exp(-sympy.Rational(5, 3))
    / (4 * sympy.pi)
    * h ** 2 / k_B
    * _m_a ** -sympy.Rational(8, 3)
    * (5 * k_B * _m - 2 * _m_a * _s)
    * _m ** -sympy.Rational(1, 3)
    * sympy.exp(sympy.Rational(2, 3) * _m_a / k_B * _s / _m),
}


def add_functionals(functionals, U, s, m, m_a, θ=None, π=None, μ=None):
    """add functionals for a distributed ideal gas
    to a dictionary of functionals.
    The functional U(s, m) will always be added.
    The functionals θ(s, m), π(s, m) and μ(s, m)
    will only be added if a symbol is provided.

    Parameters
    ----------
    functionals : dict
        A dictionary to which the functionals are added.
    U : sympy.Symbol
        Symbol for internal energy of the gas (J).
    s : sympy.Symbol
        Symbol for entropy of the gas (J/K).
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

