# prepare_spectra

Pipeline that reads KPF L1 solar spectra, shifts them to the
heliocentric frame (using the per-order barycentric velocities in the
`BARY_CORR` table — for solar data these are the heliocentric transformation),
resamples every epoch onto a common wavelength grid, and writes an `.npz` file.

The resampling scheme and science trace are selectable:

- `--method spectres` — flux-conserving resampling (default; matches `flare_pipeline.ipynb`)
- `--method cubic` — cubic-spline interpolation
- `--method frizzle` — forward-modeling resample ([frizzle](https://frizzle.readthedocs.io/en/latest/))
- `--trace SCI1|SCI2|SCI3` — a single science trace (default `SCI2`)
- `--trace ALL` — combine all three traces (see `--combine`)

When `--trace ALL`, choose how the traces are combined:

- `--combine sum` (default) — resample each trace with `--method`, then
  flux-weight sum them
- `--combine frizzle` — forward-model all three traces together in a single
  frizzle call per epoch/order (does the resample + stack at once; `--method`
  is ignored)

Output `.npz` layout: `wave (norder, npix)`,
`flux (nspec, norder, npix)`, `bjd (nspec,)`, `texp (nspec,)`,
`filename (nspec,)`.

## Environment

A `uv` environment named `solarflares` is defined in `../pyproject.toml`:

```bash
cd Solar-Flares
uv sync                        # core pipeline deps
uv sync --extra notebook       # + jupyter/matplotlib for the notebooks
```

## Usage

```bash
cd prepare_spectra
uv run python prepare_spectra.py --date 20240808 --trace SCI2 --method spectres
uv run python prepare_spectra.py --date 20240808 --trace SCI2 --method frizzle
uv run python prepare_spectra.py --date 20240808 --trace ALL  --combine sum --method cubic
uv run python prepare_spectra.py --date 20240808 --trace ALL  --combine frizzle
uv run python prepare_spectra.py --help
```

L1 files are looked up under `$DATADIR/kpf/L1/<date>` and the clear-sky
selection under `$BASEDIR/solar_csvs/kpf/<year>/<date>_socal_rvs.csv` (override
with `--l1-dir`, `--csv-dir`, `--csv`, `--glob`, or `--files`).
