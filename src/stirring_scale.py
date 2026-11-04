"""How far convective stirring moves delivered material, over the times this paper
measures.

The comparison between reconstructed delivery and present-day velocity structure
assigns a map cell to a delivery interval when it lies within 250 km of a displaced
trench point, and the water field is smoothed over 300 km. Both lengths are chosen
rather than derived, and convective stirring gives them an order of magnitude.

Richter et al. (1982) disperse 399 neutrally buoyant particles in a numerical model
of internally heated convection at Ra = 1.4e6, chosen for an upper mantle losing
5.85e-2 W m-2, and find the horizontal spread of the particles well represented by
an effective diffusivity. Their definition is the one used here, their equation 10:

    sigma^2 = kappa_e t,

where sigma is the second moment of the particle distribution about its mean
position, so sigma is the standard deviation of the displacement from the position
the reconstruction gives. The slope of sigma^2 against time gives kappa_e = 30 times
the thermal diffusivity, a dimensional 0.3 cm^2 s^-1. Steady flows do not disperse
material beyond the scale of a convection cell at all; the dispersal comes from the
time dependence of the flow, which detaches cold plumes from the upper boundary
layer at random. The same authors tested the value for scale dependence with
Richardson's distance-neighbour method and found none, which is what allows it to be
applied over the hundreds of kilometres this paper works at.

A pattern is blurred over the width of the distribution rather than its standard
deviation, so 2 sigma is tabulated beside sigma.

    python3 src/stirring_scale.py

It writes out/stirring_scale.csv and nothing else.
"""
import csv
import math
import os

import paths as P

# The two external numbers, with their source. Everything else follows from them.
KAPPA_CM2_S = 0.3           # Richter et al. (1982), their equation 11, at Ra = 1.4e6
SECONDS_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0
CM2_PER_KM2 = 1.0e10
RADII = [250.0, 300.0]      # the assignment radius and the smoothing of the water field

TIMES = [
    (25.0, 'youngest delivery interval, 8-25 Ma'),
    (29.0, 'decay time of the ridge chemical signature'),
    (37.0, 'decay time of the seismic association'),
    (50.0, 'delivery within the past 50 Myr'),
    (100.0, 'delivery within the past 100 Myr'),
    (150.0, 'youngest hydration ages of the ridge sites'),
    (380.0, 'oldest hydration age in the compilation'),
    (400.0, 'the span of the reconstruction'),
    (500.0, 'half a billion years'),
]


def main():
    kappa = KAPPA_CM2_S / CM2_PER_KM2 * SECONDS_PER_MYR      # km^2 per Myr
    print('effective diffusivity {:.1f} cm2/s = {:.0f} km2/Myr'.format(KAPPA_CM2_S, kappa))
    print('{:>7s}  {:>11s}  {:>11s}   {}'.format('t (Myr)', 'sigma', '2 sigma', 'what it is'))
    rows = []
    for t, what in TIMES:
        sigma = math.sqrt(kappa * t)
        print('{:7.0f}  {:8.0f} km  {:8.0f} km   {}'.format(t, sigma, 2 * sigma, what))
        rows.append(dict(t_Myr=t, sigma_km=round(sigma), two_sigma_km=round(2 * sigma), what=what))

    for r in RADII:
        t1, t2 = r ** 2 / kappa, (r / 2.0) ** 2 / kappa
        print('sigma reaches {:.0f} km after {:.0f} Myr, 2 sigma after {:.0f} Myr'.format(r, t1, t2))
        rows.append(dict(t_Myr=round(t1), sigma_km=round(r), two_sigma_km='',
                         what='the time at which sigma reaches {:.0f} km'.format(r)))
        rows.append(dict(t_Myr=round(t2), sigma_km='', two_sigma_km=round(r),
                         what='the time at which 2 sigma reaches {:.0f} km'.format(r)))

    out = os.path.join(P.OUT, 'stirring_scale.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['t_Myr', 'sigma_km', 'two_sigma_km', 'what'])
        w.writeheader()
        w.writerows(rows)
    print('wrote', out)


if __name__ == '__main__':
    main()
