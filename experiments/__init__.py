import os
import sys

# Expose the actual code that lives under examples/experiments as a top-level
# package so imports like `import experiments.mappings` work both inside and
# outside the repo.
_here = os.path.dirname(__file__)
_examples_pkg = os.path.join(os.path.dirname(_here), "examples", "experiments")
if _examples_pkg not in sys.path:
    sys.path.insert(0, _examples_pkg)

# Make experiments.* resolve to modules inside examples/experiments
__path__ = [_examples_pkg]
