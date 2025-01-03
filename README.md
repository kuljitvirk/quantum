# quantum
This an evolving repository of codes and notebooks that I make to learn and explore topics in solid-state physics, quantum many-body theory, information, and computation. 
My continually updated notes on [quantum computing](docs/QuantumComputing01.pdf) and [error correction](docs/QuantumComputing02.pdf).

## Contents


### `notebooks`

* [Simulation of quantum period finding algorithm](notebooks/PeriodFinding.ipynb).
* [Numerical accuracy of the time ordered exponential for a 2-level harmonically driven system](notebooks/TimeOrderedExp.ipynb).
* [Tight-binding bandstructure of 2D materials](notebooks/tmdc_bands.ipynb). 
* [Band inversion in a 2-state model.](notebooks/two-band-models.ipynb)
* [Tight binding band structure calculations for sp<sup>3</sup> bonded semiconductors](notebooks/TightBindingZB.ipynb). 

### `primitives`

* [Sherrington-Kirkpatrick model implementing an approximate optimization algorithm](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/aoa.py).
* [Functions to compute continued fraction representation of a real number and vice versa](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/mathutils.py).
* Decomposition of arbitrary unitary matrix into a product of $2\times2$ unitaries: [qprimitives.matrixutils](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/matrixutils.py).
* Class for constructing operator representations in computational basis of $n$ qubits: [qprimitives.toric](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/toric.py).
* [Period finding algorithm](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/qprimitives.py).
* [Shor's factorization algorithm](https://github.com/kuljitvirk/quantum/blob/main/qprimitives/qprimitives.py).

### `floquet`

The module [floquet.dyanmics](floquet/dynamics.py) implements the follwing functions:

* Matrix exponentials, $e^{a H}$, to construct unitary evolution operators.
* Construction of the Bloch-Pierels Hamiltonian with a harmonic vector potential.
* Direct time evolution from a periodically driven Hamiltonian by Suzuki-Trotter algorithm.
* Creation of Hamiltonian over the Bloch-Floquet upto 4th order block diagonalization.
* Computation of stroboscopic energy spectrum of a periodically driven Hamiltonian.

### `tbm`

Zinc-Blende and Diamond strucutre tight binding models for sp<sup>3</sup> bonded semiconductors: [tbm.tbzincblende](tbm/tbzincblende.py)

* Implementation from the textbook by Yu and Cardona along with the parameters of C, Si, and Ge from Chapter 2 [1].
* 10-band model with anti-bonding s-orbitals, presented in the seminal paper by Vogl et al. [2].
* [Text file](tbm/vogl_tb_parameters.txt)  with the parameter table from [2].

Tight binding models for two-dimensional materials: [tbm.tbtmdc](tbm/tbtmdc.py).
* Parameters for the tight binding model of MoS<sub>2</sub>,MoSe<sub>2</sub>,WS<sub>2</sub>,and WSe<sub>2</sub> proposed in [3]
* Tight-binding Hamiltonian of Fang et. al. decomposed by hopping vectors, so that it can be put into Bloch-Pierels form for Floquet dynamics. 


### `utils`

Plotting utilities.
 
### References

[1] Yu, Peter and Manuel Cardona, "Fundamentals of Semiconductors"

[2] Vogl, P and Hjalmarson, H and Dow, J "A SEMI-EMPIRICAL TIGHT-BINDING THEORY OF THE ELECTRONIC STRUCTURE" , I. Phys. Chom. Solids Vol. 44, No. 5. pp. 365-378, 1983. [Download](tbm/VoglPaper.pdf)

[3] Fang, S., Kuate Defo, R., Shirodkar, S. N., Lieu, S., Tritsaris, G. A., &#38; Kaxiras, E. (2015). Ab initio tight-binding Hamiltonian for transition metal dichalcogenides. Physical Review B, 92 (20). https://doi.org/10.1103/PhysRevB.92.205108
