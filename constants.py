from math import pi,sqrt

two_pi              = 2*pi
speed_light         = 299792458 # Exact in SI (NIST)
vacuum_permeability = 4e-7 * pi # Exact in SI (NIST)
vacuum_permittivity = 1./(speed_light **2 * vacuum_permeability)
vacuum_impedance    = sqrt(vacuum_permeability/vacuum_permittivity)
electron_mass       = 9.10938356e-31
planck_constant     = 6.626070040e-34
hbar                = planck_constant / two_pi
elementary_charge   = 1.6021766208e-19
magnetic_flux_quantum =  planck_constant / (2*elementary_charge)
fine_strucure_constant = elementary_charge**2 / (4*pi*vacuum_permittivity*hbar*speed_light)
bohr_radius         = 4*pi*vacuum_permittivity*hbar**2 / (electron_mass * elementary_charge**2)
boltzmann_constant  = 1.38064852e-23; # J/K
