"""Official AIRigs command-line tool.

The package talks to two services:

* an AIRigs forum (`AIRIGS_URL`) for device-flow authentication, rigs and
  hardware benchmarks;
* the Recipes platform (`RECIPES_URL`) for recipes and recipe benchmarks.

Both are documented in ``README.md``. Everything that measures hardware lives
under :mod:`airigs.bench` and degrades to ``null`` plus a note when a sensor or
optional dependency is unavailable -- it never invents a number.
"""

__version__ = "0.1.1"
