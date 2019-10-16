"""ideal gas model for distributed systems
based on the Sackur-Tetrode equation

fundamental equation: u = u(rho, s)
"""

import sympy


# state-independent constants
h = sympy.Symbol("h", real=True, positive=True)
k_B = sympy.Symbol("k_B", real=True, positive=True)
N_A = sympy.Symbol("N_A", real=True, positive=True)

# general wheight of one atom
m_atom = sympy.Symbol("m_atom", real=True, positive=True)

# wheights of atoms (from Periodic table)
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

# state variables
rho = sympy.Symbol("rho", real=True, positive=True)
s = sympy.Symbol("s", real=True, positive=True)

# fundamental equation
u = sympy.Symbol("u", real=True, positive=True)

# equations of state
T = sympy.Symbol("T", real=True, positive=True)
p = sympy.Symbol("p", real=True, positive=True)
mu = sympy.Symbol("mu", real=True, positive=True)

functionals = {
    # internal energy u(rho, s)
    u: 3
    / (4 * sympy.pi)
    * sympy.exp(-sympy.Rational(5, 3))
    * h ** 2
    * m_atom ** -sympy.Rational(8, 3)
    * rho ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * m_atom / k_B * s / rho),
    # temperature T(rho, s)
    T: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2 / k_B
    * m_atom ** -sympy.Rational(5, 3)
    * rho ** sympy.Rational(2, 3)
    * sympy.exp(sympy.Rational(2, 3) * m_atom / k_B * s / rho),
    # pressure p(rho, s)
    p: sympy.exp(-sympy.Rational(5, 3))
    / (2 * sympy.pi)
    * h ** 2
    * m_atom ** -sympy.Rational(8, 3)
    * rho ** sympy.Rational(5, 3)
    * sympy.exp(sympy.Rational(2, 3) * m_atom / k_B * s / rho),
    # chemical potential mu(rho, s)
    mu: sympy.exp(-sympy.Rational(5, 3))
    / (4 * sympy.pi)
    * h ** 2 / k_B
    * m_atom ** -sympy.Rational(8, 3)
    * (5 * k_B * rho - 2 * m_atom * s)
    * rho ** -sympy.Rational(1, 3)
    * sympy.exp(sympy.Rational(2, 3) * m_atom / k_B * s / rho),
}
