# EuropaConvection

Repository cleanup notes and working layout.

```text
EuropaConvection/
|-- docs/             Cross-project research notes, plans, and specs
|-- tools/            Repo-level utility scripts
|-- Europa2D/         Active 2D latitude-column model
|-- EuropaProjectDJ/  Nested git repo for the 1D model and thesis materials
|-- results/          Local scratch output (ignored)
`-- .claude/          Local Claude settings/worktrees
```

Key conventions:

- Keep research and planning notes under `docs/` instead of the repo root.
- Keep repo-level helper scripts under `tools/`.
- Treat `Europa2D/results`, `Europa2D/figures`, caches, LaTeX byproducts, and Claude worktrees as generated local output.
- Treat `EuropaProjectDJ` as its own project tree; its documentation now lives under `EuropaProjectDJ/docs/`.

See [docs/README.md](/c:/Users/Joshu/.cursor/projects/EuropaConvection/docs/README.md), [Europa2D/README.md](/c:/Users/Joshu/.cursor/projects/EuropaConvection/Europa2D/README.md), and [EuropaProjectDJ/README.md](/c:/Users/Joshu/.cursor/projects/EuropaConvection/EuropaProjectDJ/README.md) for the cleaned subtrees.

