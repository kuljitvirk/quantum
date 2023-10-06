# quantum
This an ever evolving repository of codes and notebooks that I make to learn and explore topics in quantum many-body, information, and computation. The explanations of the codes, their use, and technical discussion will be introduced as I complete it. 

## Contents


### `notebooks`

* Simulation of quantum period finding algorithm.
* Numerical accuracy of the time ordered exponential for a 2-level harmonically driven system. 
* Tight-binding bandstructure of 2D materials based on PRB, Fang et. al. , 95 (20), 2015.
* Band inversion in a 2-state model.

### `primitives`
* Sherrington-Kirkpatrick model implementing an approximate optimization algorithm. 
* Functions to compute continued fraction representation of a real number and vice versa.
* Decomposition of arbitrary unitary matrix into a product of $2\times2$ unitaries.
* Class for constructing operator representations in computational basis of $n$ qubits.
* Period finding algorithm.
* Shor's factorization algorithm.

### `floquet`

* Matrix exponentials, $e^{a H}$, to construct unitary evolution operators.
* Construction of the Bloch-Pierels Hamiltonian with a harmonic vector potential.
* Direct time evolution from a periodically driven Hamiltonian by Suzuki-Trotter algorithm.
* Creation of Hamiltonian over the Bloch-Floquet upto 4th order block diagonalization.
* Computation of stroboscopic energy spectrum of a periodically driven Hamiltonian.

### `tbm`

* Parameters for the tight binding model of MoS<sub>2</sub>,MoSe<sub>2</sub>,WS<sub>2</sub>,and WSe<sub>2</sub> proposed in PRB, Fang et. al. , 95 (20), 2015 arXiv: 1506.08860 [1].
* Construction of the tight-binding Hamiltonian of [1] decomposed by hopping vectors, so that it can be put into Bloch-Pierels form for Floquet dynamics. 


### `utils`

Plotting utilities.
 
