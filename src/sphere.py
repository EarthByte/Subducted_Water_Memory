# -*- coding: utf-8 -*-
"""Rotations on the sphere for the spin nulls, and the conventions that go with them.

A spin null rotates a constellation of points rigidly on the sphere, carrying its own geography
with it, so that only its alignment with the field being tested is destroyed. Two choices decide
whether such a null is honest, and both are fixed here so that every script shares them.

The rotation is drawn uniformly from SO(3) by the QR route, with the signs of the diagonal of R
folded into Q and the determinant forced positive (Mezzadri, 2007). Without the sign step the draw
is not uniform, and the obvious alternative of a uniform axis with a uniform angle is not uniform
either: the Haar angle density goes as 1 - cos(theta), so a uniform angle over-weights small
rotations, about an eighth of which leave a constellation within twenty degrees of where it was
observed and so carry the observed association into the null.

Rotations that move the constellation by less than MIN_DISPLACE_DEG in the median are therefore
rejected, which is the guard the field-scale tests in this paper already use.

These functions are duplicated from `s16_plume_null.py`, where they were written, so that the
scripts of the water inversion do not depend on an exploration script.
"""
import numpy as np

MIN_DISPLACE_DEG = 20.0


def to_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_lonlat(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (np.degrees(np.arctan2(y, x)),
            np.degrees(np.arcsin(np.clip(z / np.linalg.norm(xyz, axis=-1), -1, 1))))


def random_rotation(rng):
    """A rotation drawn uniformly from SO(3) (Mezzadri, 2007)."""
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def spun(lon, lat, rng, min_displace=MIN_DISPLACE_DEG):
    """One admissible rigid rotation of a constellation, as longitudes and latitudes."""
    xyz0 = to_xyz(np.asarray(lon, float), np.asarray(lat, float))
    while True:
        xyz = xyz0 @ random_rotation(rng).T
        disp = np.degrees(np.arccos(np.clip((xyz * xyz0).sum(1), -1, 1)))
        if np.median(disp) >= min_displace:
            return to_lonlat(xyz)


def tail(observed, null):
    """The one-sided tail probability on the side the observation falls, with the usual
    plus-one correction, as the field-scale tests in this paper report it."""
    n = np.asarray(null, float)
    n = n[np.isfinite(n)]
    if not n.size:
        return float('nan')
    if observed >= np.median(n):
        return float(((n >= observed).sum() + 1) / (n.size + 1))
    return float(((n <= observed).sum() + 1) / (n.size + 1))
