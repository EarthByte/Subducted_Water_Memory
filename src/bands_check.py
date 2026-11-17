# -*- coding: utf-8 -*-
"""Check that re-extracting the depth bands added the new band and left the others where they were.

Every published depth statistic is read from these caches, so a re-extraction has to be shown not
to have moved them. The test is the one this project already applies to a cache, correlation above
0.999 against what it is compared with, together with a difference small against the field itself:
re-running the interpolation does not reproduce the last bits, and demanding that it does would
fail on rounding while saying nothing about whether the right volume was read. A wrong source file
or a misread depth axis fails this test grossly, which is what it is for.

The band-edge table itself is expected to gain a row, and is reported rather than compared.

    python3 src/bands_check.py out/bands_REVEAL.npz <an earlier copy of the same file>
"""
import os
import sys

import numpy as np

MIN_R, MAX_REL_RMS = 0.999, 0.01

new, old = np.load(sys.argv[1]), np.load(sys.argv[2])
shared = [k for k in old.files if k in new.files and np.asarray(old[k]).ndim == 2
          and k != 'bands']
added = [k for k in new.files if k not in old.files]
bad, worst = [], (1.0, 0.0, '')
for k in shared:
    a, b = np.asarray(new[k], float), np.asarray(old[k], float)
    if a.shape != b.shape:
        bad.append('{} changed shape'.format(k))
        continue
    m = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    rms = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
    scale = float(np.sqrt(np.mean(b[m] ** 2))) or 1.0
    if r < worst[0]:
        worst = (r, rms / scale, k)
    if r < MIN_R or rms / scale > MAX_REL_RMS:
        bad.append('{}: r {:.5f}, difference {:.2%} of the field'.format(k, r, rms / scale))

print('{}: {} bands compared, {} added ({}), worst r {:.5f} at {} with a difference of {:.3%} '
      'of the field'.format(os.path.basename(sys.argv[1]), len(shared), len(added),
                            ', '.join(added) or 'none', worst[0], worst[2] or 'none', worst[1]))
if bad:
    print('  ' + '\n  '.join(bad))
    raise SystemExit('a band that existed before has moved materially; the cache cannot be trusted')
