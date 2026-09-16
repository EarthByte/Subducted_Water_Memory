#!/usr/bin/env python3
"""Download the source data for the continental intraplate volcanism catalogue.

Standard library only - no pip install, no API key. Run it from your project
folder and it writes the inputs that build_ipv_catalogue.py expects.

    python fetch_ipv_sources.py gvp                 # ~1 MB, both epochs
    python fetch_ipv_sources.py georoc --list       # see what regions exist
    python fetch_ipv_sources.py georoc --continental
    python fetch_ipv_sources.py all --then-build

WHY GEOROC NEEDS SPECIAL HANDLING
---------------------------------
The GEOROC "Intraplate Volcanic Rocks" compilation (DIGIS Team,
doi:10.25625/RZZ9VM) is 126 regional CSVs and some of them are gigabytes,
because every geochemical and isotopic column is included. We need three
columns: longitude, latitude, age. So the georoc mode STREAMS each file and
writes out only those columns as it goes - the full file is never held in
memory and never lands on disk. A multi-gigabyte region reduces to a few
hundred kilobytes, and you can run the whole compilation on a laptop.

Downloads resume: a partial GVP file is re-requested with a Range header, and
an already-reduced GEOROC region is skipped unless you pass --force.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = 'ipv-catalogue-fetcher/1.0 (research use; contact via project)'

GVP_WFS = ('https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows'
           '?service=WFS&version=1.0.0&request=GetFeature'
           '&typeName=GVP-VOTW:{layer}&outputFormat=csv')
GVP_LAYERS = {
    'holocene': 'Smithsonian_VOTW_Holocene_Volcanoes',
    'pleistocene': 'Smithsonian_VOTW_Pleistocene_Volcanoes',
}

DATAVERSE = 'https://data.goettingen-research-online.de'
GEOROC_DOI = 'doi:10.25625/RZZ9VM'

# GEOROC region files whose names indicate continental lithosphere. Used by
# --continental; everything else is oceanic islands and seamounts, which are not
# the target population. Matching is a case-insensitive substring test on the
# filename, so this is deliberately generous - check the --list output.
CONTINENTAL_HINTS = [
    'SHIELD', 'CRATON', 'FOLDBELT', 'PLATFORM', 'MASSIF', 'BASIN', 'RIFT',
    'AFRICA', 'ASIA', 'EUROPE', 'AUSTRALIA', 'ANTARCTICA', 'AMERICA',
    'CHINA', 'MONGOLIA', 'SIBERIA', 'INDIA', 'ARABIA', 'ANATOLIA', 'TURKEY',
    'IBERIA', 'PANNONIAN', 'BOHEMIA', 'RHINE', 'MASSIF_CENTRAL', 'EIFEL',
    'KOREA', 'JAPAN_SEA', 'INDOCHINA', 'MEXICO', 'PATAGONIA', 'BRAZIL',
    'COLORADO', 'RIO_GRANDE', 'BASIN_AND_RANGE', 'SNAKE_RIVER', 'YELLOWSTONE',
]


# --------------------------------------------------------------------------
def opener(insecure=False):
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def request(url, op, offset=0, timeout=120):
    req = urllib.request.Request(url, headers={
        'User-Agent': UA,
        'Accept': '*/*',
        'Accept-Encoding': 'gzip',
    })
    if offset:
        req.add_header('Range', f'bytes={offset}-')
    return op.open(req, timeout=timeout)


def body_stream(resp):
    """Transparently gunzip if the server compressed the response."""
    if resp.headers.get('Content-Encoding', '').lower() == 'gzip':
        return gzip.GzipFile(fileobj=resp)
    return resp


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return f'{n:.1f} {u}'
        n /= 1024


def download(url, dest, op, force=False, retries=4):
    """Plain download with resume and retry. Returns bytes written."""
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
        print(f'  have {os.path.basename(dest)} '
              f'({human(os.path.getsize(dest))}), skipping (use --force)')
        return os.path.getsize(dest)
    tmp = dest + '.part'
    for attempt in range(1, retries + 1):
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            resp = request(url, op, offset=offset)
            total = resp.headers.get('Content-Length')
            total = (int(total) + offset) if total else None
            mode = 'ab' if offset and resp.status == 206 else 'wb'
            if mode == 'wb':
                offset = 0
            src = body_stream(resp)
            done = offset
            t0 = time.time()
            with open(tmp, mode) as fh:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if time.time() - t0 > 2:
                        pct = f' {100 * done / total:.0f}%' if total else ''
                        print(f'\r  {os.path.basename(dest)}: '
                              f'{human(done)}{pct}   ', end='', flush=True)
                        t0 = time.time()
            print(f'\r  {os.path.basename(dest)}: {human(done)}          ')
            os.replace(tmp, dest)
            return done
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError) as e:
            print(f'\n  attempt {attempt}/{retries} failed: {e}')
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


# --------------------------------------------------------------------------
def fetch_gvp(outdir, op, force, epochs):
    written = []
    for epoch in epochs:
        url = GVP_WFS.format(layer=GVP_LAYERS[epoch])
        dest = os.path.join(outdir, f'gvp_{epoch}.csv')
        print(f'GVP {epoch}')
        try:
            download(url, dest, op, force=force)
        except Exception as e:
            print(f'  ! {e}')
            print('  fall back to the Excel export at '
                  'https://volcano.si.edu/gvp_votw.cfm and save it as CSV')
            continue
        # verify
        with open(dest, encoding='utf-8-sig', errors='replace') as fh:
            head = fh.readline()
            n = sum(1 for _ in fh)
        ok = 'Tectonic_Setting' in head or 'Tectonic Setting' in head
        print(f'  {n} rows, tectonic-setting column '
              f'{"present" if ok else "MISSING - check the file"}')
        if not ok:
            print(f'  first line was: {head[:200]}')
        written.append(dest)
    return written


# --------------------------------------------------------------------------
def dataverse_files(op):
    url = (f'{DATAVERSE}/api/datasets/:persistentId/?persistentId='
           f'{urllib.parse.quote(GEOROC_DOI)}')
    with request(url, op) as resp:
        meta = json.load(body_stream(resp))
    files = meta['data']['latestVersion']['files']
    out = []
    for f in files:
        df = f['dataFile']
        out.append(dict(id=df['id'], name=df.get('filename', ''),
                        size=int(df.get('filesize', 0))))
    return sorted(out, key=lambda d: d['name'])


GEOROC_KEEP = ('LONGITUDE', 'LATITUDE', 'AGE', 'LOCATION', 'MATERIAL',
               'ROCK NAME', 'ROCK_NAME', 'TECTONIC SETTING', 'TECTONIC_SETTING')


def reduce_georoc(fileinfo, outdir, op, force):
    """Stream one GEOROC regional CSV and keep only the columns we need."""
    dest = os.path.join(outdir, 'reduced_' + fileinfo['name'])
    if os.path.exists(dest) and not force:
        print(f'  have {os.path.basename(dest)}, skipping')
        return dest
    url = f'{DATAVERSE}/api/access/datafile/{fileinfo["id"]}'
    print(f"  {fileinfo['name']} ({human(fileinfo['size'])}) -> reducing")

    with request(url, op, timeout=300) as resp:
        stream = io.TextIOWrapper(body_stream(resp), encoding='latin-1',
                                  errors='replace', newline='')
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            print('   ! empty file')
            return None
        upper = [h.strip().upper() for h in header]

        def find(*needles):
            for i, h in enumerate(upper):
                if all(n in h for n in needles):
                    return i
            return None

        idx = {}
        for label, args in (('lon', ('LONGITUDE', 'MIN')),
                            ('lat', ('LATITUDE', 'MIN')),
                            ('age', ('AGE', 'MIN')),
                            ('location', ('LOCATION',)),
                            ('setting', ('TECTONIC',)),
                            ('rock', ('ROCK', 'NAME'))):
            j = find(*args)
            if j is None and label in ('lon', 'lat', 'age'):
                j = find(args[0])
            if j is not None:
                idx[label] = j
        if 'lon' not in idx or 'lat' not in idx:
            print(f'   ! no coordinate columns; header began '
                  f'{header[:6]}')
            return None

        cols = list(idx)
        n_in = n_out = 0
        with open(dest, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for row in reader:
                n_in += 1
                if not row or len(row) <= max(idx.values()):
                    # the precompiled files end with a references block whose
                    # rows are shorter than the data rows - stop there
                    if row and row[0].strip().lower().startswith('reference'):
                        break
                    continue
                vals = [row[idx[c]].strip() for c in cols]
                if not vals[cols.index('lon')] or not vals[cols.index('lat')]:
                    continue
                w.writerow(vals)
                n_out += 1
    print(f'   {n_in} rows read, {n_out} kept, '
          f'{human(os.path.getsize(dest))} on disk')
    return dest


def fetch_georoc(outdir, op, force, list_only, continental, regions, max_mb):
    files = dataverse_files(op)
    total = sum(f['size'] for f in files)
    print(f'GEOROC intraplate compilation: {len(files)} files, '
          f'{human(total)} total')
    if list_only:
        for f in files:
            print(f"  {human(f['size']):>10}  {f['name']}")
        return []

    sel = files
    if regions:
        pats = [r.upper() for r in regions]
        sel = [f for f in sel if any(p in f['name'].upper() for p in pats)]
    elif continental:
        sel = [f for f in sel
               if any(h in f['name'].upper() for h in CONTINENTAL_HINTS)]
        dropped = [f['name'] for f in files if f not in sel]
        if dropped:
            # a name-based filter is a guess, so say exactly what it threw away
            print(f'  --continental excluded {len(dropped)} region file(s):')
            for d in dropped:
                print(f'      {d}')
            print('    If a continental region you want is in that list, rerun '
                  'without --continental\n    (streaming makes the full '
                  'compilation cheap) or name it with --regions.')
    if max_mb:
        skipped = [f for f in sel if f['size'] > max_mb * 1024 * 1024]
        sel = [f for f in sel if f['size'] <= max_mb * 1024 * 1024]
        if skipped:
            print(f'  ! skipping {len(skipped)} file(s) over {max_mb} MB: '
                  + ', '.join(s['name'] for s in skipped[:5])
                  + (' ...' if len(skipped) > 5 else ''))
            print('    (streaming means size is not really a problem - '
                  'raise --max-mb if you want them)')

    print(f'selected {len(sel)} of {len(files)} files, '
          f'{human(sum(f["size"] for f in sel))} to stream')
    out = []
    for i, f in enumerate(sel, 1):
        print(f'[{i}/{len(sel)}]')
        try:
            d = reduce_georoc(f, outdir, op, force)
            if d:
                out.append(d)
        except Exception as e:
            print(f'   ! failed: {e}')
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Download source data for the IPV catalogue.')
    ap.add_argument('what', choices=['gvp', 'georoc', 'all'])
    ap.add_argument('--outdir', default='ipv_sources')
    ap.add_argument('--force', action='store_true', help='re-download')
    ap.add_argument('--insecure', action='store_true',
                    help='skip TLS verification (some mirrors have bad certs)')
    ap.add_argument('--epochs', nargs='+', default=['holocene', 'pleistocene'],
                    choices=['holocene', 'pleistocene'])
    ap.add_argument('--list', action='store_true',
                    help='georoc: list the regional files and exit')
    ap.add_argument('--continental', action='store_true',
                    help='georoc: only regions whose name looks continental')
    ap.add_argument('--regions', nargs='+',
                    help='georoc: substring match on filenames, e.g. CHINA SIBERIA')
    ap.add_argument('--max-mb', type=float, default=0,
                    help='georoc: skip files larger than this (0 = no limit)')
    ap.add_argument('--then-build', action='store_true',
                    help='run build_ipv_catalogue.py on what was downloaded')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    op = opener(a.insecure)
    gvp_files, georoc_files = [], []

    if a.what in ('gvp', 'all'):
        gvp_files = fetch_gvp(a.outdir, op, a.force, a.epochs)
        print()
    if a.what in ('georoc', 'all'):
        georoc_files = fetch_georoc(a.outdir, op, a.force, a.list,
                                    a.continental, a.regions, a.max_mb)
        print()

    if a.list:
        return 0

    print('downloaded:')
    for f in gvp_files + georoc_files:
        print(f'  {f}')
    if not (gvp_files or georoc_files):
        print('  nothing - see the messages above')
        return 1

    builder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'build_ipv_catalogue.py')
    if a.then_build and os.path.exists(builder):
        import subprocess
        if gvp_files:
            cmd = [sys.executable, builder, '--source', 'gvp', *gvp_files,
                   '--out', 'ipv_catalogue_gvp.csv']
            print('\n$ ' + ' '.join(cmd))
            subprocess.run(cmd, check=False)
        if georoc_files:
            cmd = [sys.executable, builder, '--source', 'georoc', *georoc_files,
                   '--max-age', '250', '--out', 'ipv_catalogue_georoc.csv']
            print('\n$ ' + ' '.join(cmd))
            subprocess.run(cmd, check=False)
    else:
        print('\nnext:')
        if gvp_files:
            print(f'  python build_ipv_catalogue.py --source gvp '
                  f'{" ".join(gvp_files)} --out ipv_catalogue_gvp.csv')
        if georoc_files:
            print(f'  python build_ipv_catalogue.py --source georoc '
                  f'"{a.outdir}/reduced_*.csv" --max-age 250 '
                  f'--out ipv_catalogue_georoc.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
