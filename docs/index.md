# dcrpy

`dcrpy` provides tools for Doppler Cloud Radar reading, retrieval, plotting, and
dual-frequency analysis.

## Documentation Structure

- Single-frequency binary access through `dcrpy.rpg_binary.rpg`
- Dual-frequency alignment and DFR analysis through `dcrpy.dual_rpg.dual_rpg`
- Retrieval formulas centralized in `dcrpy.retrieve.retrieve`

## Build Locally

Install the documentation extras:

```bash
pip install -e ".[docs]"
```

Then serve the site locally:

```bash
mkdocs serve
```

Or build the static site:

```bash
mkdocs build
```
