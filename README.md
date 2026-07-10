# Solar-Flares
Research project 2025

## ./prepare_spectra

Command-line pipeline that reads KPF L1 solar spectra, shifts them to the
heliocentric frame, and resamples onto a common wavelength grid (schemes:
`spectres` / `cubic` / `frizzle`). See
[`prepare_spectra/README.md`](prepare_spectra/README.md) for details and usage.

A `uv` environment named `solarflares` is defined in `pyproject.toml`
(`uv sync`).
