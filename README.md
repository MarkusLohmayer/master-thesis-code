# Thesis title:  Towards a port-Hamiltonian approach to study Stirling-cycle devices

This repo contains the Python code and the Jupyter notebooks
which were used to produce the simulation results
shown in my master's thesis which I wrote in 2019 at FAU Erlangen.

More information about this research on
port-Hamiltonian systems
and Stirling-cycle devices can be found on
my [personal research website](https://markuslohmayer.github.io/research/).

The code folder contains the Python files
which are imported from the Jupyter notebooks.

The notebook `butcher.ipynb`
was used to learn how to compute Butcher tableaus
for Gauss-Legendre collocation methods.

The notebooks
`harmonic_oscillator.ipynb`,
`kepler.ipynb` and
`spring_pendulum.ipynb`
deal with conservative mechanical systems
and merely serve to test the implementation of the
Gauss-Legendre collocation methods.

The notebooks
`piston_animation_euler.ipynb`,
`piston_animation.ipynb` and
`piston_animation2.ipynb`
deal with the central example of the thesis.
The first one just uses the explicit Euler scheme.
The other two use Gauss-Legendre collocation methods
and rely on automated code generation using SymPy.

The notebook `ideal_gas_sym.ipynb`
is supposed to check
the equations which describe an ideal gas.

The notebook `piston.ipynb` contains some symbolic computations
concerning the modeling of the piston example
which I also could have done on paper.

The notebook `carnot_efficiency.ipynb`
simply plots the Carnot efficiency for different temperatures.
