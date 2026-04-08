# Europa2D

Axisymmetric/latitude-column Europa shell model.

```text
Europa2D/
|-- docs/      Design notes and validation writeups
|-- src/       Model and sampling code
|-- scripts/   Run, plotting, and table-generation entry points
|-- tests/     Unit and validation tests
|-- results/   Generated Monte Carlo archives
`-- figures/   Generated plots and poster/table assets
```

Working rules:

- Treat `results/` as generated data.
- Treat plotted `png`/`pdf` outputs in `figures/` as generated products.
- Keep source assets such as `figures/*.tex` and all docs tracked.

