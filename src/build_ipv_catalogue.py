"""Build a continental intraplate volcanism catalogue for the wet-plume tests.

WHY THIS SCRIPT EXISTS RATHER THAN A HARDCODED LIST
---------------------------------------------------
The target population is the one Wang et al. (2025) used: onshore volcanism
classified as intraplate, away from plate boundaries. Coordinates for that
population have to come from a citable source, not from anyone's recollection,
so this script ingests a source file and does the filtering and aggregation
reproducibly. Two sources are supported:

  --source gvp     Smithsonian Global Volcanism Program, Volcanoes of the World.
                   Filters on the Tectonic Setting field. Holocene and/or
                   Pleistocene. Small, authoritative, one file.

  --source georoc  GEOROC "Intraplate Volcanic Rocks" precompiled compilation
                   (DIGIS Team, doi:10.25625/RZZ9VM) — the exact dataset behind
                   Wang et al. Pass one or more regional CSVs; only the
                   longitude, latitude and age columns are read, so the huge
                   geochemistry columns are streamed past rather than loaded.

TWO DESIGN DECISIONS WORTH ARGUING WITH
---------------------------------------
GVP separates "Intraplate / Continental crust (>25 km)" from "Rift zone /
Continental crust (>25 km)". By default only the former is kept, because Wang et
al. define intraplate as away from plate boundaries and an active rift is a
plate boundary. `--include-rift` relaxes this; the East African Rift is the
population that moves.

Nearby vents are NOT independent observations. A volcanic field with forty cones
would otherwise contribute forty points to a spin test that assumes one per
site — the same pseudo-replication that inflated the LIP analysis by a factor of
seven before provinces were introduced. Points are therefore agglomerated by
single linkage on the sphere at `--cluster-km` (default 100 km) and each cluster
contributes one entry at its centroid. `--cluster-km 0` disables it, and the
report prints how many raw points collapsed into each field so the effect is
visible rather than silent.

Output is the format plume_classifier.py and s17_wet_plume.py expect —
`hotspot, lat, lon_180` — plus provenance columns.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

R_EARTH = 6371.0

GVP_INTRAPLATE = 'Intraplate / Continental crust (>25 km)'
GVP_RIFT = 'Rift zone / Continental crust (>25 km)'


def to_xyz(lon, lat):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def wrap180(lon):
    return ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0


# --------------------------------------------------------------------------
def read_gvp(path, include_rift, include_oceanic):
    """GVP exports vary: comma or semicolon, and sometimes two preamble lines."""
    df = None
    for sep in (',', ';', '\t'):
        for skip in (0, 1, 2):
            try:
                t = pd.read_csv(path, sep=sep, skiprows=skip, dtype=str,
                                encoding='utf-8-sig', engine='python')
            except Exception:
                continue
            cols = {c.strip().lower(): c for c in t.columns}
            if any('tectonic' in c for c in cols) and any('latitude' in c for c in cols):
                df = t
                break
        if df is not None:
            break
    if df is None:
        # GVP's Pleistocene table genuinely has no Tectonic_Setting column, so
        # this is an expected outcome rather than a parse failure. Say which it
        # is instead of dying with a generic message.
        try:
            probe = pd.read_csv(path, nrows=0, encoding='utf-8-sig',
                                engine='python')
            have = list(probe.columns)
        except Exception:
            have = None
        if have is not None and not any('tectonic' in c.lower() for c in have):
            print(f'  ! {os.path.basename(path)} has no tectonic-setting column '
                  f'and cannot be filtered by setting - SKIPPED')
            print(f'    (columns: {", ".join(have[:14])}'
                  f'{" ..." if len(have) > 14 else ""})')
            print('    GVP publishes the setting field for the Holocene table '
                  'only; the Pleistocene table omits it.')
            return pd.DataFrame(columns=['hotspot', 'lat', 'lon_180',
                                         'tectonic_setting', 'source_id',
                                         'country', 'rock_type', 'age_Ma',
                                         'source'])
        raise SystemExit(
            f'could not parse {path} as a GVP export: no Tectonic Setting + '
            f'Latitude columns found under any of , ; tab with 0-2 preamble rows')

    def col(*needles):
        for c in df.columns:
            lc = c.strip().lower()
            if all(n in lc for n in needles):
                return c
        return None

    c_set = col('tectonic')
    c_lat, c_lon = col('latitude'), col('longitude')
    c_name = col('volcano', 'name') or col('name')
    c_num = col('volcano', 'number')
    c_country = col('country')
    c_rock = col('rock')

    # GVP is not internally consistent about this string: the live web service
    # writes "Intraplate / Continental crust (> 25 km)" and some exports write
    # "(>25 km)". Matching the literal would silently return zero rows, so
    # whitespace is collapsed and the match is on the two words that carry the
    # meaning rather than on the full label.
    settings = (df[c_set].fillna('').astype(str)
                .str.replace(r'\s+', ' ', regex=True).str.strip())
    print('  tectonic settings present in this file:')
    for k, v in settings.value_counts().items():
        print(f'    {v:5d}  {k or "(blank)"}')

    is_intra = settings.str.startswith('Intraplate')
    is_rift = settings.str.startswith('Rift zone')
    is_cont = settings.str.contains('Continental', case=False)

    keep = is_intra & is_cont
    if include_rift:
        keep |= is_rift & is_cont
    if include_oceanic:
        keep |= is_intra
    sel = df[keep].copy()
    print(f'  kept {int(keep.sum())} rows'
          + ('' if keep.any() else '  <-- NOTHING MATCHED, see the list above'))

    out = pd.DataFrame({
        'hotspot': sel[c_name].astype(str).str.strip() if c_name else 'unnamed',
        'lat': pd.to_numeric(sel[c_lat], errors='coerce'),
        'lon_180': wrap180(pd.to_numeric(sel[c_lon], errors='coerce')),
        'tectonic_setting': settings[keep].values,
        'source_id': sel[c_num].astype(str) if c_num else '',
        'country': sel[c_country].astype(str) if c_country else '',
        'rock_type': sel[c_rock].astype(str) if c_rock else '',
        'age_Ma': np.nan,
        'source': 'GVP',
    })
    return out.dropna(subset=['lat', 'lon_180'])


# GEOROC's "Intraplate Volcanic Rocks" compilation is classified per sample and
# leaks material that is neither intraplate nor volcanic, concentrated in the
# North American Cordillera files: the Coast Plutonic Complex, the Coast
# Mountains Batholith, the Spences Bridge Group, the Huerto Andesite. Those are
# arc and plutonic units, and they sit near the Cascadia slab, so the noise runs
# in the same direction as the effect being tested. Two cheap filters remove
# most of it: a catalogue of VOLCANISM should not contain plutonic rock names,
# and a catalogue of volcanic FIELDS should not contain sample-site codes.
PLUTONIC_ROCK = ('GRANITE', 'GRANODIORITE', 'TONALITE', 'DIORITE', 'GABBRO',
                 'MONZONITE', 'SYENITE', 'PEGMATITE', 'ANORTHOSITE',
                 'CHARNOCKITE', 'PLUTONIC', 'BATHOLITH', 'INTRUSION',
                 'INTRUSIVE', 'PERIDOTITE', 'DUNITE', 'PYROXENITE', 'ECLOGITE',
                 'AMPHIBOLITE', 'GNEISS', 'SCHIST', 'XENOLITH')
NON_FIELD_NAME = ('BATHOLITH', 'PLUTONIC COMPLEX', 'PLUTON', 'SITE ', 'DREDGE',
                  'DSDP', 'ODP ', 'IODP', 'BOREHOLE', 'DRILL')


def georoc_age_Ma(series):
    """GEOROC ages are in YEARS, tagged with a citation id, sometimes several.

    A raw value looks like '54000000 [21571]', or '14000000 [7809] / 0 [8533]'
    when two sources disagree. Feeding that to pd.to_numeric returns NaN for
    every row, which is exactly what happened the first time: 1210 real ages in
    the Australia file alone were silently dropped and the whole catalogue came
    out ageless. The first numeric token is taken and converted to Ma.
    """
    s = series.astype(str).str.strip()
    first = s.str.extract(r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', expand=False)
    return pd.to_numeric(first, errors='coerce') / 1.0e6


def read_georoc(paths, max_age, keep_undated=False, drop_plutonic=True):
    """Read only the coordinate and age columns from the precompiled CSVs."""
    frames = []
    for p in paths:
        head = pd.read_csv(p, nrows=0, encoding='latin-1', engine='python',
                           on_bad_lines='skip')
        cols = {c.strip().upper(): c for c in head.columns}

        def pick(*needles):
            for u, orig in cols.items():
                if all(n in u for n in needles):
                    return orig
            return None

        # accepts both the raw precompiled headers ("LONGITUDE (MIN.)") and the
        # short headers written by fetch_ipv_sources.py's streaming reducer
        c_lon = pick('LONGITUDE', 'MIN') or pick('LONGITUDE') or pick('LON')
        c_lat = pick('LATITUDE', 'MIN') or pick('LATITUDE') or pick('LAT')
        c_age = pick('AGE', 'MIN') or pick('AGE')
        c_loc = pick('LOCATION')
        c_rock = pick('ROCK', 'NAME') or pick('ROCK')
        if not (c_lon and c_lat):
            print(f'  ! {os.path.basename(p)}: no coordinate columns, skipped')
            continue
        use = [c for c in (c_lon, c_lat, c_age, c_loc, c_rock) if c]
        t = pd.read_csv(p, usecols=use, encoding='latin-1', engine='python',
                        on_bad_lines='skip', dtype=str)
        t = t.rename(columns={c_lon: 'lon_180', c_lat: 'lat'})
        t['lon_180'] = wrap180(pd.to_numeric(t['lon_180'], errors='coerce'))
        t['lat'] = pd.to_numeric(t['lat'], errors='coerce')
        t['age_Ma'] = georoc_age_Ma(t[c_age]) if c_age else np.nan
        t['_rock'] = t[c_rock].astype(str).str.upper() if c_rock else ''
        t['_loc'] = t[c_loc].astype(str).str.upper() if c_loc else ''
        t['hotspot'] = (t[c_loc].astype(str).str.split('/').str[-1].str.strip()
                        if c_loc else os.path.basename(p))
        t['source'] = 'GEOROC:' + os.path.basename(p)
        frames.append(t[['hotspot', 'lat', 'lon_180', 'age_Ma', 'source',
                         '_rock', '_loc']])
        print(f'  {os.path.basename(p)}: {len(t)} rows')
    if not frames:
        raise SystemExit('no usable GEOROC files')
    df = pd.concat(frames, ignore_index=True).dropna(subset=['lat', 'lon_180'])
    if drop_plutonic:
        n0 = len(df)
        bad_rock = df['_rock'].str.contains('|'.join(PLUTONIC_ROCK), na=False)
        bad_name = df['_loc'].str.contains('|'.join(
            n.replace(' ', r'\s*') for n in NON_FIELD_NAME), na=False, regex=True)
        drop = bad_rock | bad_name
        if drop.any():
            worst = (df.loc[drop, 'hotspot'].astype(str).str.title()
                     .value_counts().head(6))
            print(f'  plutonic / non-field filter: {n0} -> {n0 - int(drop.sum())} '
                  f'rows ({int(bad_rock.sum())} by rock name, '
                  f'{int(bad_name.sum())} by unit name)')
            print('    most removed: ' + ', '.join(
                f'{k} ({v})' for k, v in worst.items()))
        df = df[~drop]
    df = df.drop(columns=['_rock', '_loc'])
    if max_age is not None:
        n0 = len(df)
        n_undated = int(df.age_Ma.isna().sum())
        if keep_undated:
            df = df[df.age_Ma.isna() | (df.age_Ma <= max_age)]
            print(f'  age filter <= {max_age} Ma: {n0} -> {len(df)} rows '
                  f'({n_undated} undated rows KEPT via --keep-undated)')
            print('    ! undated GEOROC samples include Proterozoic and Archean '
                  'material.\n      Comparing that to present-day tomography is '
                  'not meaningful; prefer the default.')
        else:
            df = df[df.age_Ma.notna() & (df.age_Ma <= max_age)]
            print(f'  age filter <= {max_age} Ma: {n0} -> {len(df)} rows '
                  f'({n_undated} undated rows dropped; --keep-undated retains them)')
    df['tectonic_setting'] = 'GEOROC intraplate compilation'
    df['source_id'] = ''
    df['country'] = ''
    df['rock_type'] = ''
    return df


# --------------------------------------------------------------------------
def _labels_numpy(lon, lat, radius_km):
    """Single-linkage labels using numpy only, via spatial bucketing + union-find.

    scipy is not always present (the plain system python on a Mac usually has
    numpy and pandas but not scipy), and an all-pairs distance matrix is not an
    option once GEOROC contributes tens of thousands of samples anyway. Points
    are binned into cells about one linkage radius across and only same-cell and
    adjacent-cell pairs are tested, which is what makes this linear in practice.
    """
    n = len(lon)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Bucket in 3-D on the unit sphere rather than in lon/lat. A lon/lat grid
    # has two failure modes that cost real pairs: cells stop being adjacent
    # across the +/-180 seam, and their east-west extent collapses toward the
    # poles. Cubes of side equal to the chord subtending the linkage radius have
    # neither, and any two points within the radius differ by at most one cube
    # index on each axis, so the 27-neighbourhood is exhaustive.
    xyz = to_xyz(lon, lat)
    side = 2.0 * np.sin(radius_km / (2.0 * R_EARTH))
    idx = np.floor(xyz / side).astype(int)

    buckets = {}
    for k, key in enumerate(map(tuple, idx.tolist())):
        buckets.setdefault(key, []).append(k)

    cos_r = np.cos(radius_km / R_EARTH)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
               for c in (-1, 0, 1)]
    for key, members in buckets.items():
        near = []
        for o in offsets:
            near.extend(buckets.get((key[0] + o[0], key[1] + o[1],
                                     key[2] + o[2]), ()))
        near = np.array(near, dtype=int)
        if not len(near):
            continue
        for k in members:
            d = xyz[near] @ xyz[k]
            for m in near[d >= cos_r]:
                if m != k:
                    union(k, int(m))
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab + 1


def _labels_grid(lon, lat, radius_km):
    """One field per occupied cell of a fixed spherical grid.

    Single linkage is the wrong algorithm once sampling is dense. GEOROC has
    168 000 samples, so at 100 km almost every point is within the radius of
    some other point, the clusters chain, and the first run of this collapsed
    38% of the entire compilation into ONE 'field' spanning continents. Grid
    binning cannot chain: cell membership depends only on where a point is, not
    on what else was sampled nearby. Cubes on the unit sphere are used rather
    than a lon/lat mesh so cells do not shrink toward the poles or break at the
    dateline.
    """
    xyz = to_xyz(lon, lat)
    side = 2.0 * np.sin(radius_km / (2.0 * R_EARTH))
    idx = np.floor(xyz / side).astype(np.int64)
    # a stable integer key per cell
    keys = (idx[:, 0].astype(object).astype(str) + '_'
            + idx[:, 1].astype(object).astype(str) + '_'
            + idx[:, 2].astype(object).astype(str))
    _, lab = np.unique(keys, return_inverse=True)
    return lab + 1


def _base_name(name):
    """Strip the disambiguation suffixes this script itself adds."""
    import re
    n = str(name).strip()
    n = re.sub(r'\s+field(\s+\d+)?$', '', n, flags=re.I)
    n = re.sub(r'\s+\d+$', '', n)
    return n.upper()


def merge_same_province(cat, radius_km, max_span_km=400.0):
    """Re-join grid cells that are pieces of one named province.

    Grid binning cures chaining but introduces the opposite failure: a province
    larger than a cell is split into several "fields" - the Coast Plutonic
    Complex came out as five, the Navajo Volcanic Field as three - which inflates
    n with points that are anything but independent. Cells are merged when they
    share a base name AND are close enough to be one province, so two unrelated
    features that happen to carry a generic name are not welded together.
    """
    if len(cat) < 2:
        return cat
    base = cat.hotspot.map(_base_name)
    parent = np.arange(len(cat))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    xyz = to_xyz(cat.lon_180.values.astype(float), cat.lat.values.astype(float))
    cos_lim = np.cos(min(max_span_km, 6.0 * radius_km) / R_EARTH)
    for _, idx in base.groupby(base).groups.items():
        ii = np.asarray([cat.index.get_loc(i) for i in idx], int)
        if len(ii) < 2:
            continue
        for a in range(len(ii)):
            d = xyz[ii] @ xyz[ii[a]]
            for b in np.where(d >= cos_lim)[0]:
                ra, rb = find(int(ii[a])), find(int(ii[b]))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(cat))])

    # GEOROC's location field is often an administrative region rather than a
    # volcanic field - "Tibet", "Inner Mongolia", "Shandong Province" - so a
    # pure name merge welds thousands of kilometres into one object. Any group
    # whose diameter exceeds max_span_km is therefore dissolved back into its
    # grid cells: names are trusted only where the geography agrees with them.
    oversize = 0
    cos_span = np.cos(max_span_km / R_EARTH)
    for r in np.unique(roots):
        ii = np.where(roots == r)[0]
        if len(ii) < 2:
            continue
        d = xyz[ii] @ xyz[ii].T
        if d.min() < cos_span:
            roots[ii] = ii          # dissolve: every cell becomes its own group
            oversize += 1
    if oversize:
        print(f'  province merge: {oversize} name group(s) exceeded '
              f'{max_span_km:.0f} km and were left as grid cells')
    if len(np.unique(roots)) == len(cat):
        return cat

    rows = []
    for r in np.unique(roots):
        g = cat.iloc[roots == r]
        w = g.n_points.values.astype(float)
        v = (to_xyz(g.lon_180.values.astype(float), g.lat.values.astype(float))
             * w[:, None]).sum(0)
        v /= np.linalg.norm(v)
        rec = g.iloc[0].to_dict()
        rec.update(
            hotspot=_base_name(g.iloc[0].hotspot).title(),
            lat=float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))),
            lon_180=float(np.degrees(np.arctan2(v[1], v[0]))),
            n_points=int(w.sum()),
            n_cells=int(len(g)),
            age_Ma=(float(np.average(g.age_Ma.values, weights=w))
                    if g.age_Ma.notna().all() else g.age_Ma.median()))
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(f'  province merge: {len(cat)} cells -> {len(out)} fields '
          f'(largest province spanned {int(out.n_cells.max())} cells)')
    return out


def cluster(df, radius_km, method='auto'):
    """Aggregate raw sample points into volcanic fields."""
    if radius_km <= 0 or len(df) < 2:
        df = df.copy()
        df['n_points'] = 1
        return df
    if method == 'auto':
        method = 'grid' if len(df) > 2000 else 'linkage'
    lon = df.lon_180.values.astype(float)
    lat = df.lat.values.astype(float)
    if method == 'grid':
        lab = _labels_grid(lon, lat, radius_km)
    else:
        lab = _labels_numpy(lon, lat, radius_km)
    print(f'  aggregation method: {method}')

    rows = []
    for c in np.unique(lab):
        g = df[lab == c]
        v = to_xyz(g.lon_180.values, g.lat.values).mean(0)
        v /= np.linalg.norm(v)
        names = [n for n in g.hotspot.astype(str) if n and n.lower() != 'nan']
        name = (pd.Series(names).mode().iloc[0] if names else 'unnamed')
        rows.append(dict(
            hotspot=name if len(g) == 1 else f'{name} field',
            lat=float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))),
            lon_180=float(np.degrees(np.arctan2(v[1], v[0]))),
            n_points=int(len(g)),
            age_Ma=float(g.age_Ma.median()) if g.age_Ma.notna().any() else np.nan,
            tectonic_setting=g.tectonic_setting.iloc[0],
            country=g.country.iloc[0],
            rock_type=g.rock_type.iloc[0],
            source=g.source.iloc[0],
            source_id=g.source_id.iloc[0]))
    out = pd.DataFrame(rows)
    # make names unique - the downstream merge is on hotspot
    dup = out.hotspot.duplicated(keep=False)
    if dup.any():
        out.loc[dup, 'hotspot'] = (out.loc[dup, 'hotspot'] + ' ' +
                                   (out.loc[dup].groupby('hotspot').cumcount() + 1
                                    ).astype(str))
    return out


TYPE_EXAMPLES = ['Changbaishan', 'Baitoushan', 'Wudalianchi', 'Hainan', 'Leizhou',
                 'Tengchong', 'Jeju', 'Ulleung', 'Eifel', 'Chaine des Puys',
                 'Massif Central', 'Yellowstone']


def main():
    ap = argparse.ArgumentParser(
        description='Build a continental intraplate volcanism catalogue.')
    ap.add_argument('--source', choices=['gvp', 'georoc'], required=True)
    ap.add_argument('files', nargs='+', help='input CSV(s), or a glob for GEOROC')
    ap.add_argument('--out', default='ipv_catalogue.csv')
    ap.add_argument('--cluster-km', type=float, default=100.0,
                    help='aggregate points at roughly this scale into one field '
                         '(0 disables)')
    ap.add_argument('--method', default='auto',
                    choices=['auto', 'grid', 'linkage'],
                    help="'grid' bins to a fixed spherical mesh and cannot "
                         "chain; 'linkage' is single-linkage agglomeration, "
                         "correct only for sparse catalogues; 'auto' picks grid "
                         "above 2000 points")
    ap.add_argument('--max-age', type=float, default=250.0,
                    help='GEOROC only: keep samples younger than this, Ma')
    ap.add_argument('--no-province-merge', action='store_true',
                    help='skip re-joining grid cells that share a province name')
    ap.add_argument('--keep-plutonic', action='store_true',
                    help='GEOROC only: keep plutonic rock names and batholith / '
                         'sample-site entries, which the compilation mislabels '
                         'as intraplate volcanism')
    ap.add_argument('--keep-undated', action='store_true',
                    help='GEOROC only: retain samples with no age. Most of the '
                         'compilation is undated and much of it is Precambrian, '
                         'so this is off by default')
    ap.add_argument('--include-rift', action='store_true',
                    help='GVP only: also keep "Rift zone / Continental crust", '
                         'which adds the East African Rift')
    ap.add_argument('--include-oceanic', action='store_true',
                    help='GVP only: also keep intraplate oceanic crust')
    a = ap.parse_args()

    paths = []
    for f in a.files:
        paths.extend(sorted(glob.glob(f)) or [f])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit('not found: ' + ', '.join(missing))

    print(f'reading {len(paths)} file(s) as {a.source}')
    if a.source == 'gvp':
        parts = [read_gvp(p, a.include_rift, a.include_oceanic) for p in paths]
        parts = [p for p in parts if len(p)]
        if not parts:
            raise SystemExit(
                'no usable rows. If you passed only the Pleistocene table, note '
                'it carries no tectonic-setting field - use the Holocene table.')
        df = pd.concat(parts, ignore_index=True)
    else:
        df = read_georoc(paths, a.max_age, a.keep_undated,
                         not a.keep_plutonic)

    print(f'\n{len(df)} points pass the setting filter')
    cat = cluster(df, a.cluster_km, a.method)
    if not a.no_province_merge:
        cat = merge_same_province(cat, a.cluster_km)
    big = int(cat.n_points.max())
    print(f'{len(cat)} fields at {a.cluster_km:.0f} km '
          f'(largest holds {big} points = {100 * big / cat.n_points.sum():.1f}% '
          f'of the sample; {int((cat.n_points > 1).sum())} fields are multi-point)')
    if big > 0.10 * cat.n_points.sum():
        print('  ! one field holds more than a tenth of the whole compilation. '
              'That is chaining,\n    not geography - use --method grid or a '
              'smaller --cluster-km.')

    cat = cat.sort_values('hotspot').reset_index(drop=True)
    # names must be unique: everything downstream merges on 'hotspot'
    dup = cat.hotspot.duplicated(keep=False)
    if dup.any():
        cat.loc[dup, 'hotspot'] = (
            cat.loc[dup, 'hotspot'].astype(str) + ' ('
            + cat.loc[dup, 'lat'].round(1).astype(str) + ', '
            + cat.loc[dup, 'lon_180'].round(1).astype(str) + ')')
        print(f'  {int(dup.sum())} duplicate names disambiguated by coordinate')
    cols = ['hotspot', 'lat', 'lon_180', 'n_points', 'n_cells', 'age_Ma',
            'tectonic_setting', 'country', 'rock_type', 'source', 'source_id']
    cat[[c for c in cols if c in cat]].to_csv(a.out, index=False)

    # ------------------------------------------------------------------ QC
    print('\nQC')
    print(f'  latitude range  {cat.lat.min():+.1f} to {cat.lat.max():+.1f}')
    print(f'  longitude range {cat.lon_180.min():+.1f} to {cat.lon_180.max():+.1f}')
    bad = cat[(cat.lat.abs() > 90) | (cat.lon_180.abs() > 180)]
    print(f'  out-of-range coordinates: {len(bad)}')
    print(f'  duplicate names: {int(cat.hotspot.duplicated().sum())}')
    if 'age_Ma' in cat:
        na = cat.age_Ma.notna().sum()
        print(f'  ages: {na}/{len(cat)} fields dated'
              + (f', median {cat.age_Ma.median():.1f} Ma, '
                 f'{int((cat.age_Ma <= 25).sum())} younger than 25 Ma'
                 if na else '  <-- NONE DATED, check the age column parsing'))
    if 'country' in cat and cat.country.astype(str).str.len().max() > 0:
        top = cat.country.value_counts().head(8)
        print('  top countries: ' + ', '.join(f'{k} {v}' for k, v in top.items()))
    hits = [t for t in TYPE_EXAMPLES
            if cat.hotspot.astype(str).str.contains(t, case=False, na=False).any()]
    print(f'  wet-plume type examples present ({len(hits)}/{len(TYPE_EXAMPLES)}): '
          f'{", ".join(hits) if hits else "NONE - check the source coverage"}')
    if len(cat) < 40:
        print(f'  ! only {len(cat)} fields; a spin test needs enough points to '
              f'resolve a correlation. Consider --cluster-km smaller, '
              f'--include-rift, or a broader source.')

    print(f'\nwrote {a.out} ({len(cat)} fields)')
    print('run it through the wet-plume tests with:')
    print(f'  python s17_wet_plume.py --file REVEAL_vs_full.nc --var voigt \\\n'
          f'      --hotspots {a.out} --summary <plume_summary csv>')


if __name__ == '__main__':
    main()
