"""Present-day and reconstructed plate boundaries drawn the way a tectonic map
draws them: spreading ridges as thin lines beneath the data symbols, subduction
zones as a line with teeth pointing towards the overriding plate.

gplately returns a tessellated trench as one row per sample point, and the
eighth column of that row is the trench normal azimuth, measured clockwise from
north and pointing in the direction of subduction (pygplates' convention). A
tooth is a triangle whose base lies along the trench and whose apex sits that
way, so the teeth need no polarity flag of their own.

    import plate_boundaries as PB
    PB.ridges(ax, recon, 0.0, geo)
    PB.trenches(ax, recon, 0.0, geo)

Both take whatever `PlateReconstruction` the figure already built, so the model
is the figure's model, and both fail quietly: a time outside the model's range
leaves the map without boundaries rather than without a figure.
"""
import numpy as np

R = 6371.0

RIDGE_COLOUR = '#1D6BAA'
TRENCH_COLOUR = '#1a1a1a'


def _offset(lon, lat, azimuth, arc_deg):
    """Great-circle offset of a point by arc_deg along azimuth (deg from north)."""
    d = np.radians(arc_deg)
    th = np.radians(azimuth)
    p1 = np.radians(lat)
    l1 = np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(l2), np.degrees(p2)


def ridges(ax, recon, time, geo, colour=RIDGE_COLOUR, lw=0.6, zorder=3, alpha=0.9):
    """Spreading ridges as a thin line. Drawn point by point, so a ridge that
    crosses the date line does not close across the map."""
    try:
        mor = recon.tessellate_mid_ocean_ridges(
            float(time), tessellation_threshold_radians=0.005, ignore_warnings=True)
    except Exception as e:                      # a time outside the model, say
        print(f'  no ridges at {time:.0f} Ma: {e}')
        return
    if mor is None or not len(mor):
        return
    ax.scatter(mor[:, 0], mor[:, 1], s=lw * 1.2, c=colour, lw=0, alpha=alpha,
               transform=geo, zorder=zorder)


def trenches(ax, recon, time, geo, colour=TRENCH_COLOUR, lw=0.7, zorder=3,
             tooth_deg=1.6, spacing_deg=6.0):
    """Subduction zones as a line with teeth towards the overriding plate.

    tooth_deg is the height of a tooth in degrees of arc and spacing_deg the
    distance between teeth along the trench; both scale with the map, so a
    regional panel wants smaller values than a global one.
    """
    try:
        sz = recon.tessellate_subduction_zones(
            float(time), tessellation_threshold_radians=0.005, ignore_warnings=True)
    except Exception as e:
        print(f'  no trenches at {time:.0f} Ma: {e}')
        return
    if sz is None or not len(sz):
        return
    lon, lat = sz[:, 0], sz[:, 1]
    ax.scatter(lon, lat, s=lw * 1.4, c=colour, lw=0, transform=geo, zorder=zorder)
    if sz.shape[1] < 8:                         # no azimuth column: line only
        return
    az = sz[:, 7]

    # one tooth every spacing_deg of arc along the trench, following the order
    # in which the points come back; a jump means a new trench segment
    step = np.zeros(len(lon))
    step[1:] = np.degrees(np.arccos(np.clip(
        np.sin(np.radians(lat[:-1])) * np.sin(np.radians(lat[1:])) +
        np.cos(np.radians(lat[:-1])) * np.cos(np.radians(lat[1:])) *
        np.cos(np.radians(lon[1:] - lon[:-1])), -1, 1)))
    step[step > spacing_deg] = 0.0              # segment break, start again
    run, take = 0.0, []
    for i, s in enumerate(step):
        run += s
        if i == 0 or run >= spacing_deg:
            take.append(i)
            run = 0.0
    tri = []
    for i in take:
        # near the poles a fixed angular offset spans a huge range of longitude
        # and the triangles smear into a comb, so the line is left bare there
        if abs(lat[i]) > 72.0:
            continue
        apex = _offset(lon[i], lat[i], az[i], tooth_deg)
        left = _offset(lon[i], lat[i], az[i] - 90.0, tooth_deg * 0.55)
        right = _offset(lon[i], lat[i], az[i] + 90.0, tooth_deg * 0.55)
        # a tooth that straddles the date line is dropped rather than drawn
        # across the whole map
        xs = [left[0], apex[0], right[0]]
        if max(xs) - min(xs) > 180.0:
            continue
        tri.append([left, apex, right])
    for t in tri:
        ax.fill([p[0] for p in t], [p[1] for p in t], facecolor=colour,
                edgecolor='none', transform=geo, zorder=zorder)
    print(f'  {len(tri)} teeth on {len(lon)} trench points at {time:.0f} Ma')
