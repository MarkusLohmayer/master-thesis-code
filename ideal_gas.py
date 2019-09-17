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
m = M / N_A


C1 = 2.5 + 1.5 * log((4.0 * pi * m) / (3.0 * h**2))



"""entropy as a thermodynamic potential"""

def S(U, V, N):
    """Sackur-Tetrode equation"""
    return N * k_B * (log((V / N) * (U / N)**(3/2)) + C1)

def S_1_T(U, V, N):
    """inverse temperature"""
    return (3 * k_B * N) / (2 * U)

def S_T(U, V, N):
    """temperature"""
    return (2 * U) / (3 * k_B * N)

def S_p_T(U, V, N):
    """pressure / temperature"""
    return N * k_B / V

def S_p(U, V, N):
    """pressure"""
    return (2 * U) / (3 * V)

def S_mu_T(U, V, N):
    """chemical potential / temperature"""
    return -k_B*(C1 + log(V*(U/N)**1.5/N)) - 2.5*k_B

def S_mu(U, V, N):
    """chemical potential"""
    return -2/3*U*(k_B*(C1 + log(V*(U/N)**1.5/N)) - 2.5*k_B)/(N*k_B)



"""internal energy as a thermodynamic potential"""

def U(S, V, N):
    """Sackur-Tetrode equation solved for U"""
    return N * ((N / V) * exp(S / (k_B * N) - C1))**(2/3)

def U2(T, N):
    """internial energy as a function of T, V, N"""
    return 1.5 * k_B * N * T

C2 = h**2 / (2 * pi * exp(5/3) * m)
C3 = C2 / k_B

def U_T(S, V, N):
    """temperature"""
    return C3 * (N * exp(S / (N * k_B)) / V)**(2/3)

def U_p(S, V, N):
    """pressure"""
    return C2 * (N / V) * (N * exp(S / (N * k_B)) / V)**(2/3)

def U_mu(S, V, N):
    """chemical potential"""
    return (1/3)*(5.0*N*k_B - 2.0*S)*exp((2/3)*(-C1*N*k_B + S)/(N*k_B))/(N**(1/3)*V**(2/3)*k_B)


"""Helmholtz free energy as a thermodynamic potential"""

def F(T, V, N):
    """Hemholtz free energy"""
    return 0.5*N*T*k_B*(-2*C1 + log((8/27)*N**2/(T**3*V**2*k_B**3)) + 3)

def F_S(T, V, N):
    """entropy"""
    return 0.5*N*k_B*(-2.0*C1 + log((8/27)*N**2/(T**3*V**2*k_B**3)))

def F_p(T, V, N):
    """pressure"""
    return k_B * N * T / V

def F_mu(T, V, N):
    """chemical potential"""
    return 0.5*T*k_B*(-2.0*C1 + log((8/27)*N**2/(T**3*V**2*k_B**3)) + 5)

