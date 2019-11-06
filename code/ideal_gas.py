"""ideal gas model based on the Sackur-Tetrode equation"""

from math import log, exp, pi


# Boltzmann's constant
k_B = 1.380649e-23 # J / K

# Planck's constant
h = 6.62607015e-34 # J s

# Avogadro constant
N_A = 6.02214076 * 10**23 # 1 / mol

# Wheight of one mole of Argon
M = 39.88 / 1000 # kg / mol

# mass of one atom in kg
m_a = M / N_A


C1 = 2.5 + 1.5 * log((4.0 * pi * m_a) / (3.0 * h**2))


"""entropy as a thermodynamic potential"""

def S(u, v, n):
    """Sackur-Tetrode equation"""
    return n * k_B * (log((v / n) * (u / n)**(3/2)) + C1)

def S_1_θ(u, v, n):
    """inverse temperature"""
    return (3 * k_B * n) / (2 * u)

def S_θ(u, v, n):
    """temperature"""
    return (2 * u) / (3 * k_B * n)

def S_π_θ(u, v, n):
    """pressure / temperature"""
    return n * k_B / v

def S_π(u, v, n):
    """pressure"""
    return (2 * u) / (3 * v)

def S_μ_θ(u, v, n):
    """chemical potential / temperature"""
    return -k_B*(C1 + log(v*(u/n)**1.5/n)) - 2.5*k_B

def S_μ(u, v, n):
    """chemical potential"""
    return -2/3*u*(k_B*(C1 + log(v*(u/n)**1.5/n)) - 2.5*k_B)/(n*k_B)



"""internal energy as a thermodynamic potential"""

def U(s, v, n):
    """Sackur-Tetrode equation solved for U"""
    return n * ((n / v) * exp(s / (k_B * n) - C1))**(2/3)

def U2(θ, n):
    """internal energy as a function of θ, n"""
    return 1.5 * k_B * n * θ

C2 = h**2 / (2 * pi * exp(5/3) * m_a)
C3 = C2 / k_B

def U_θ(s, v, n):
    """temperature"""
    return C3 * (n * exp(s / (n * k_B)) / v)**(2/3)

def U_π(s, v, n):
    """pressure"""
    return C2 * (n / v) * (n * exp(s / (n * k_B)) / v)**(2/3)

def U_μ(s, v, n):
    """chemical potential"""
    return (1/3)*(5.0*n*k_B - 2.0*s)*exp((2/3)*(-C1*n*k_B + s)/(n*k_B))/(n**(1/3)*v**(2/3)*k_B)


"""Helmholtz free energy as a thermodynamic potential"""

def F(θ, v, n):
    """Hemholtz free energy"""
    return 0.5*n*θ*k_B*(-2*C1 + log((8/27)*n**2/(θ**3*v**2*k_B**3)) + 3)

def F_s(θ, v, n):
    """entropy"""
    return -0.5*n*k_B*(-2.0*C1 + log((8/27)*n**2/(θ**3*v**2*k_B**3)))

def F_π(θ, v, n):
    """pressure"""
    return k_B * n * θ / v

def F_μ(θ, v, n):
    """chemical potential"""
    return 0.5*θ*k_B*(-2.0*C1 + log((8/27)*n**2/(θ**3*v**2*k_B**3)) + 5)
