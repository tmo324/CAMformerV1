# Structure

This standalone static companion follows Tergel's project-structure skill (`/Users/tmolomochir/.agents/skills/project-structure/SKILL.md`). The upstream research repository is unchanged.

- `index.html`: semantic interface and explanatory copy.
- `p00_data/`: immutable exported release-model snapshot.
- `p01_ui/`: page controller, pure attention example and responsive styles.
- `p02_vendor/`: Plotly browser bundle and its license.
- `p03_tools/`: upstream-model exporter and resolved Python dependency lock.
- `p04_tests/`: exhaustive teaching-model correctness checks.
- `p05_figures/`: original upstream figure images for comparison.

Root HTML/favicon are kept at the static-host entry point. No Node package environment is required; the app has no npm dependencies. The Python lock is only needed to regenerate the data.
