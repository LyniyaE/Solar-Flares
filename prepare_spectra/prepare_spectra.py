#!/usr/bin/env python3
"""
prepare_spectra.py
==================

Command-line pipeline that turns KPF L1 solar spectra into a time series of
spectra sampled on a common (heliocentric-frame) wavelength grid.

The steps mirror the ``flare_pipeline.ipynb`` / ``flare_frizzle.ipynb`` notebooks:

  1. Read the KPF L1 spectra (per trace / order / pixel), in the observatory frame.
  2. Shift each spectrum to the heliocentric frame using the per-order barycentric
     velocity stored in the ``BARY_CORR`` table (for these solar data the
     "barycentric" quantities are in fact the heliocentric transformation).
  3. Resample every spectrum onto a single common wavelength grid using one of
     several interchangeable schemes (``--method``).
  4. Optionally combine the three science traces (``--trace ALL``).
  5. Write an ``.npz`` file with the same layout as the notebooks:
        wave     (norder, npix)             common wavelength grid
        flux     (nspec, norder, npix)      resampled flux
        bjd      (nspec,)                    barycentric/heliocentric JD
        texp     (nspec,)                    exposure time [s]
        filename (nspec,)                    source L1 file paths

The resampling scheme is fully modular: pick with ``--method``
(``spectres`` | ``cubic`` | ``frizzle``) and the trace with ``--trace``
(``SCI1`` | ``SCI2`` | ``SCI3`` | ``ALL``; default ``SCI2``).

Example
-------
    uv run python prepare_spectra.py --date 20240808 --trace SCI2 --method spectres
    uv run python prepare_spectra.py --date 20240808 --trace SCI2 --method frizzle
    uv run python prepare_spectra.py --date 20240808 --trace ALL  --method cubic
"""

import os
import sys
import glob
import argparse

import numpy as np


SPEED_OF_LIGHT = 299792458.0  # m/s

# KPF cameras, in the order the notebooks stack them into the order axis
# (GREEN orders first, then RED). Wavelength/flux arrays are flipped in the
# dispersion direction so wavelength is monotonically increasing.
CAMERAS = ("GREEN", "RED")
NTRACE = 3  # SCI1, SCI2, SCI3

TRACE_INDEX = {"SCI1": 0, "SCI2": 1, "SCI3": 2}

# Default data locations follow the flare_frizzle.ipynb convention: the L1 files
# live under ``$DATADIR/kpf/L1/<date>`` and the SoCal RV CSVs under
# ``$BASEDIR/solar_csvs/kpf/<year>/<date>_socal_rvs.csv``. These env vars are set
# in the user's shell (e.g. DATADIR=~/ceph/data, BASEDIR=~/ceph); the CLI flags
# override them.
DEFAULT_DATADIR = os.environ.get("DATADIR", "/mnt/home/rrubenzahl/ceph/data")
DEFAULT_BASEDIR = os.environ.get("BASEDIR", "/mnt/home/rrubenzahl/ceph")
DEFAULT_L1_DIR  = os.path.join(DEFAULT_DATADIR, "kpf", "L1")
DEFAULT_CSV_DIR = os.path.join(DEFAULT_BASEDIR, "solar_csvs")


# ---------------------------------------------------------------------------
# 1. Reading L1 spectra
# ---------------------------------------------------------------------------
def read_l1_spectra(l1files, getblaze=False, verbose=True):
    """
    Read a list of KPF L1 files into dense arrays.

    Returns a dict of arrays with shape ``(nspec, ntrace, norder, npix)`` for the
    per-pixel quantities:

        waves  : wavelength solution [Angstrom] (increasing)
        fluxs  : extracted flux [e-]
        vars   : flux variance
        blaze  : blaze function (ones if ``getblaze`` is False)

    and per-spectrum / per-order quantities:

        bcvels : (nspec, norder) barycentric (== heliocentric) velocity [m/s]
        bjds   : (nspec,)        barycentric/heliocentric JD
        texps  : (nspec,)        exposure time [s]
        files  : (nspec,)        the L1 file paths actually read
    """
    from astropy.io import fits
    from astropy.table import Table

    waves, fluxs, varss, blazes = [], [], [], []
    bcvels, bjds, texps, files = [], [], [], []

    for k, l1file in enumerate(l1files):
        if not os.path.exists(l1file):
            if verbose:
                print(f"  [skip] missing file: {l1file}")
            continue

        wave_t, flux_t, var_t, blaze_t = [], [], [], []
        with fits.open(l1file) as hdu:
            for trace in range(1, NTRACE + 1):  # SCI1, SCI2, SCI3
                w_all, f_all, v_all, b_all = [], [], [], []
                for camera in CAMERAS:
                    # flip dispersion axis so wavelength increases
                    w = hdu[f"{camera}_SCI_WAVE{trace}"].data[:, ::-1]
                    f = hdu[f"{camera}_SCI_FLUX{trace}"].data[:, ::-1]
                    v = hdu[f"{camera}_SCI_VAR{trace}"].data[:, ::-1]
                    if getblaze:
                        b = hdu[f"{camera}_SCI_BLAZE{trace}"].data[:, ::-1]
                    else:
                        b = np.ones_like(f)

                    # Replace isolated NaNs (known bad pixels near pix 1018-1021)
                    for o in range(len(f)):
                        nan = np.isnan(f[o])
                        if np.any(nan):
                            f[o][nan] = (
                                np.nanmean(f[o][1015:1017])
                                + np.nanmean(f[o][1022:1024])
                            ) / 2.0

                    w_all.extend(w)
                    f_all.extend(f)
                    v_all.extend(v)
                    b_all.extend(b)

                wave_t.append(w_all)
                flux_t.append(f_all)
                var_t.append(v_all)
                blaze_t.append(b_all)

            # Per-order barycentric velocity (== heliocentric shift for solar data)
            bctable = Table(hdu["BARY_CORR"].data)
            bcvel = np.asarray(bctable["BARYVEL"], dtype=float)  # (norder,), m/s
            texp = float(hdu[0].header["ELAPSED"])

            # Barycentric/heliocentric JD: prefer the L2 CCFBJD (as in the main
            # notebook); fall back to the per-order PHOTON_BJD in BARY_CORR.
            bjd = _get_bjd(l1file, bctable)

        waves.append(wave_t)
        fluxs.append(flux_t)
        varss.append(var_t)
        blazes.append(blaze_t)
        bcvels.append(bcvel)
        bjds.append(bjd)
        texps.append(texp)
        files.append(l1file)

        if verbose:
            print(f"  read {k + 1}/{len(l1files)}: {os.path.basename(l1file)}")

    if not files:
        raise RuntimeError("No L1 files could be read.")

    return dict(
        waves=np.array(waves),
        fluxs=np.array(fluxs),
        vars=np.array(varss),
        blaze=np.array(blazes),
        bcvels=np.array(bcvels),
        bjds=np.array(bjds),
        texps=np.array(texps),
        files=np.array(files),
    )


def _get_bjd(l1file, bctable):
    """Best-effort barycentric/heliocentric JD for one exposure."""
    from astropy.io import fits

    l2file = l1file.replace("L1", "L2")
    if os.path.exists(l2file):
        try:
            with fits.open(l2file) as l2hdu:
                return float(l2hdu[0].header["CCFBJD"])
        except (KeyError, OSError):
            pass
    # Fall back to the photon-weighted BJD stored per order in BARY_CORR.
    for col in ("PHOTON_BJD", "PHOTONBJD"):
        if col in bctable.colnames:
            return float(np.nanmedian(np.asarray(bctable[col], dtype=float)))
    return np.nan


# ---------------------------------------------------------------------------
# 2. Normalization (shared pre-step, keeps output flux scale consistent)
# ---------------------------------------------------------------------------
def normalize_orders(flux, var, percentile=95.0):
    """
    Normalize each order by a high percentile of its flux, matching the main
    notebook (``fluxs / nanpercentile(flux, 95)``). Operates on the last axis.

    ``flux`` / ``var`` have shape (..., npix). Returns normalized copies.
    """
    norm = np.nanpercentile(flux, percentile, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return flux / norm, var / norm**2


# ---------------------------------------------------------------------------
# 3. Resampling schemes  (all share the signature below)
#
#     resample(w_out, w, f, v) -> (flux_out, err_out)
#
#   w_out : target wavelength grid (increasing)
#   w,f,v : input wavelength / flux / variance for one order (already BC-shifted)
#
# Pixels of the output grid that fall outside the input range are returned as
# NaN so the shared edge cleanup can fill them.
# ---------------------------------------------------------------------------
def resample_spectres(w_out, w, f, v):
    import spectres

    idx = np.argsort(w)
    flux_out, err_out = spectres.spectres(
        w_out, w[idx], f[idx], spec_errs=np.sqrt(v[idx]),
        fill=np.nan, verbose=False,
    )
    return np.asarray(flux_out), np.asarray(err_out)


def resample_cubic(w_out, w, f, v):
    from scipy.interpolate import CubicSpline

    idx = np.argsort(w)
    ws, fs, vs = w[idx], f[idx], v[idx]
    cs = CubicSpline(ws, fs, extrapolate=False)  # NaN outside the input range
    flux_out = cs(w_out)
    # Simple variance propagation: linear interpolation of the input variance.
    var_out = np.interp(w_out, ws, vs, left=np.nan, right=np.nan)
    return np.asarray(flux_out), np.sqrt(var_out)


def make_frizzle_resampler(n_modes=None):
    """
    Build a frizzle-based resampler. ``n_modes`` defaults to ~npix/3 (rounded to
    an odd number), matching the ~3 pixels-per-resolution-element sampling of KPF.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import frizzle

    def resample_frizzle(w_out, w, f, v):
        idx = np.argsort(w)
        nm = n_modes if n_modes is not None else (int(len(w_out) // 3) | 1)
        y_star, C_star, _flags, _meta = frizzle.frizzle(
            np.asarray(w_out), np.asarray(w[idx]),
            np.asarray(f[idx]), np.asarray(1.0 / v[idx]),
            n_modes=nm,
        )
        y_star = np.asarray(y_star)
        C_star = np.asarray(C_star)
        var = np.diag(C_star) if C_star.ndim == 2 else C_star
        # C_star is +inf in no-data regions and can pick up tiny negative values
        # from round-off; clip so the error is real (inf stays inf -> inf err).
        return y_star, np.sqrt(np.where(var < 0, 0.0, var))

    return resample_frizzle


METHOD_NAMES = ("spectres", "cubic", "frizzle")


def make_resampler(method, n_modes=None):
    """
    Return a ``resample(w_out, w, f, v)`` callable for the named ``method``.

    Built from a plain string (not a closure), so worker processes can
    reconstruct the resampler locally rather than pickling it -- important for
    ``frizzle``, whose resampler closes over an imported ``jax``/``frizzle``.
    """
    if method == "spectres":
        return resample_spectres
    if method == "cubic":
        return resample_cubic
    if method == "frizzle":
        return make_frizzle_resampler(n_modes)
    raise ValueError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# 4. Edge cleanup (shared post-step; mirrors do_bc in the main notebook)
# ---------------------------------------------------------------------------
def fill_edge_nans(flux):
    """
    Replace NaNs in a 1-D resampled order (typically at the grid edges, where the
    BC-shifted input does not cover the output grid) with the median of the
    nearest five valid pixels on the appropriate side.
    """
    isnan = np.where(np.isnan(flux))[0]
    if len(isnan) == 0:
        return flux
    n = len(flux)
    lo = np.nanmedian(flux[:5])
    hi = np.nanmedian(flux[-5:])
    for i in isnan:
        flux[i] = lo if i < n / 2 else hi
    return flux


# ---------------------------------------------------------------------------
# 5. Core: shift to heliocentric frame + resample onto the common grid
#
# Each order is resampled independently, so orders can be processed in parallel.
# The heavy loops all funnel through the array-level helpers below; the trace
# drivers fan them out over ``nproc`` worker processes (a batch of nproc orders
# runs at a time), or run serially in-process when ``nproc <= 1``.
# ---------------------------------------------------------------------------
def _resample_order_arrays(w, f, v, bcvel, w_out, resample):
    """
    Shift + resample one order across every epoch.

    ``w``, ``f``, ``v`` have shape (nspec, npix_in); ``bcvel`` has shape
    (nspec,); ``w_out`` has shape (npix_out,). Returns (nspec, npix_out) arrays.
    """
    nspec, npix = f.shape[0], len(w_out)
    flux_out = np.empty((nspec, npix))
    err_out = np.empty((nspec, npix))
    for j in range(nspec):
        # Shift to heliocentric frame: lambda_helio = lambda * (1 + v_bc/c)
        w_shift = w[j] * (1.0 + bcvel[j] / SPEED_OF_LIGHT)
        fo, eo = resample(w_out, w_shift, f[j], v[j])
        flux_out[j] = fill_edge_nans(fo)
        err_out[j] = eo
    return flux_out, err_out


def _frizzle_stack_arrays(w, f, v, bcvel, w_out, n_modes=None):
    """
    Combine all traces of one order across every epoch by forward-modeling them
    together with frizzle (a single frizzle call per epoch over the concatenated,
    BC-shifted trace pixels).

    ``w``, ``f``, ``v`` have shape (nspec, ntrace, npix_in); ``bcvel`` (nspec,).
    Returns (nspec, npix_out) arrays.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import frizzle

    nspec, ntrace = f.shape[0], f.shape[1]
    npix = len(w_out)
    nm = n_modes if n_modes is not None else (int(npix // 3) | 1)
    w_out = np.asarray(w_out)

    flux_out = np.empty((nspec, npix))
    err_out = np.empty((nspec, npix))
    for j in range(nspec):
        ws, fs, ivs = [], [], []
        for tr in range(ntrace):
            # Same per-order BC velocity for every trace (BARYVEL is per order).
            ws.append(w[j, tr] * (1.0 + bcvel[j] / SPEED_OF_LIGHT))
            fs.append(f[j, tr])
            ivs.append(1.0 / v[j, tr])
        W = np.hstack(ws)
        F = np.hstack(fs)
        IV = np.hstack(ivs)
        idx = np.argsort(W)
        y_star, C_star, _flags, _meta = frizzle.frizzle(
            w_out, np.asarray(W[idx]), np.asarray(F[idx]),
            np.asarray(IV[idx]), n_modes=nm,
        )
        y_star = np.asarray(y_star)
        C_star = np.asarray(C_star)
        var = np.diag(C_star) if C_star.ndim == 2 else C_star
        flux_out[j] = fill_edge_nans(y_star)
        err_out[j] = np.sqrt(np.where(var < 0, 0.0, var))
    return flux_out, err_out


# -- worker functions (module level so they are picklable for multiprocessing) --
def _resample_order_job(job):
    resample = make_resampler(job["method"], job["n_modes"])
    fo, eo = _resample_order_arrays(
        job["w"], job["f"], job["v"], job["bcvel"], job["w_out"], resample
    )
    return job["o"], fo, eo


def _frizzle_stack_job(job):
    fo, eo = _frizzle_stack_arrays(
        job["w"], job["f"], job["v"], job["bcvel"], job["w_out"], job["n_modes"]
    )
    return job["o"], fo, eo


def _map_orders(jobs, worker, nproc, verbose, label):
    """
    Run ``worker`` on each per-order job, yielding results as they finish.

    ``nproc <= 1`` runs serially in-process; otherwise a spawn-based process
    pool processes the orders ``nproc`` at a time. Results may arrive out of
    order, so each carries its order index ``o``.
    """
    n = len(jobs)
    if nproc is None or nproc <= 1:
        for k, job in enumerate(jobs):
            res = worker(job)
            if verbose:
                print(f"  {label} {k + 1}/{n}")
            yield res
        return

    import multiprocessing as mp

    # spawn (not fork): avoids fork-after-jax-init hangs in the frizzle workers.
    # Each worker is pinned to a single BLAS/XLA thread (see _init_worker): the
    # parallelism is across processes, so without this the nproc workers each
    # grab every core and thrash -- making frizzle *slower* than serial.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            if verbose:
                print(f"  {label} {k + 1}/{n} (nproc={nproc})")
            yield res


def _init_worker():
    """
    Pin a pool worker to a single thread so ``nproc`` processes don't oversubscribe
    the cores. Runs once per worker at startup, before any jax/BLAS work.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    # XLA reads this when the CPU client initializes (before the first frizzle
    # call in this worker, since jax is imported lazily inside the job).
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    ).strip()
    # numpy's BLAS is already loaded by the time this runs; limit it at runtime.
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def shift_and_resample_order(data, t, o, resample, w_out=None):
    """
    BC-shift a single trace/order to the heliocentric frame across every epoch
    and resample onto the output grid ``w_out`` using the ``resample`` callable.

    If ``w_out`` is None, the first-epoch wavelengths of this trace/order are
    used (sorted increasing). Returns arrays (nspec, npix): flux_out, err_out.
    """
    if w_out is None:
        w_out = np.sort(data["waves"][0, t, o])
    return _resample_order_arrays(
        data["waves"][:, t, o], data["fluxs"][:, t, o], data["vars"][:, t, o],
        data["bcvels"][:, o], w_out, resample,
    )


def shift_and_resample_trace(data, t, method, common_grid, n_modes=None,
                             nproc=1, verbose=True):
    """
    For a single trace ``t``, BC-shift every epoch/order to the heliocentric
    frame and resample onto ``common_grid`` (shape (norder, npix)) using the
    named ``method``. Orders are processed in parallel over ``nproc`` workers.

    Returns arrays (nspec, norder, npix): flux_out, err_out.
    """
    nspec, _, norder, npix = data["fluxs"].shape
    flux_out = np.empty((nspec, norder, npix))
    err_out = np.empty((nspec, norder, npix))

    jobs = [
        dict(o=o, method=method, n_modes=n_modes,
             w=data["waves"][:, t, o], f=data["fluxs"][:, t, o],
             v=data["vars"][:, t, o], bcvel=data["bcvels"][:, o],
             w_out=common_grid[o])
        for o in range(norder)
    ]
    for o, fo, eo in _map_orders(jobs, _resample_order_job, nproc, verbose,
                                 "resampled order"):
        flux_out[:, o] = fo
        err_out[:, o] = eo
    return flux_out, err_out


def combine_traces(flux_traces, err_traces):
    """
    Flux-weighted combination of the science traces onto the common grid,
    matching the main notebook. ``flux_traces`` is (ntrace, norder, npix) for a
    single epoch; returns (norder, npix).
    """
    flux_traces = np.asarray(flux_traces)
    # integrated flux per trace per order -> weights normalized across traces
    weights = np.nansum(flux_traces, axis=-1)          # (ntrace, norder)
    wsum = np.nansum(weights, axis=0)                  # (norder,)
    wsum = np.where(wsum == 0, np.nan, wsum)
    weights = weights / wsum
    totflux = np.nansum(
        [nf.T * w for nf, w in zip(flux_traces, weights)], axis=0
    ).T
    # errors added in quadrature with the same weights
    if err_traces is not None:
        err_traces = np.asarray(err_traces)
        toterr = np.sqrt(
            np.nansum(
                [(ef.T * w) ** 2 for ef, w in zip(err_traces, weights)], axis=0
            )
        ).T
    else:
        toterr = None
    return totflux, toterr


def frizzle_stack_traces(data, o, w_out, n_modes=None):
    """
    Combine all science traces onto ``w_out`` for one order by forward-modeling
    them together with frizzle. For each epoch, the BC-shifted pixels of
    SCI1/SCI2/SCI3 are concatenated and passed to a single frizzle call, so the
    resampling and the trace combination happen in one step (unlike
    ``combine_traces``, which resamples each trace separately and then sums).

    Returns arrays (nspec, npix): flux_out, err_out.
    """
    return _frizzle_stack_arrays(
        data["waves"][:, :, o], data["fluxs"][:, :, o], data["vars"][:, :, o],
        data["bcvels"][:, o], w_out, n_modes=n_modes,
    )


def frizzle_stack_trace_all_orders(data, common_grid, n_modes=None,
                                   nproc=1, verbose=True):
    """
    Frizzle-stack all traces for every order (see :func:`frizzle_stack_traces`).
    Orders are processed in parallel over ``nproc`` workers. Returns
    (nspec, norder, npix): flux_out, err_out.
    """
    nspec, _, norder, npix = data["fluxs"].shape
    flux_out = np.empty((nspec, norder, npix))
    err_out = np.empty((nspec, norder, npix))

    jobs = [
        dict(o=o, n_modes=n_modes,
             w=data["waves"][:, :, o], f=data["fluxs"][:, :, o],
             v=data["vars"][:, :, o], bcvel=data["bcvels"][:, o],
             w_out=common_grid[o])
        for o in range(norder)
    ]
    for o, fo, eo in _map_orders(jobs, _frizzle_stack_job, nproc, verbose,
                                 "frizzle-stacked order"):
        flux_out[:, o] = fo
        err_out[:, o] = eo
    return flux_out, err_out


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def resolve_l1_files(args):
    """Return the list of L1 file paths to process."""
    basepath = os.path.join(args.l1_dir, args.date)

    if args.files:
        return list(args.files)

    if args.glob:
        files = sorted(glob.glob(os.path.join(basepath, "KP.*_L1.fits")))
        if not files:
            raise RuntimeError(f"No L1 files matched in {basepath}")
        return files

    # Default: use the SoCal RV CSV to select (clear-sky) spectra.
    import pandas as pd

    csv = args.csv or os.path.join(
        args.csv_dir, "kpf", args.date[:4], f"{args.date}_socal_rvs.csv"
    )
    if not os.path.exists(csv):
        raise RuntimeError(
            f"RV CSV not found: {csv}\n"
            "Pass --csv, or use --glob to discover files directly, "
            "or --files to list them explicitly."
        )
    socal = pd.read_csv(csv)
    if (not args.all_sky) and ("clearsky" in socal.columns):
        sel = socal["clearsky"].values.astype(bool)
        print(f"  {np.sum(sel)}/{len(sel)} clear-sky spectra selected")
        rows = socal["filename"].values[sel]
    else:
        rows = socal["filename"].values

    files = []
    for fn in rows:
        l1name = os.path.basename(str(fn)).replace("L2", "L1")
        files.append(os.path.join(basepath, l1name))
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--date", required=True, help="UT date, YYYYMMDD (e.g. 20240808)")
    p.add_argument(
        "--method", default="spectres", choices=sorted(METHOD_NAMES),
        help="resampling scheme (default: spectres)",
    )
    p.add_argument(
        "--trace", default="SCI2", choices=["SCI1", "SCI2", "SCI3", "ALL"],
        help="science trace to process; ALL combines the three traces "
             "(see --combine) (default: SCI2)",
    )
    p.add_argument(
        "--combine", default="sum", choices=["sum", "frizzle"],
        help="how to combine traces when --trace ALL: 'sum' resamples each "
             "trace with --method then flux-weight sums them; 'frizzle' "
             "forward-models all three traces together in a single frizzle "
             "call (ignores --method) (default: sum)",
    )

    # File discovery
    p.add_argument(
        "--l1-dir", default=DEFAULT_L1_DIR,
        help=f"base dir holding per-date L1 folders "
             f"(default: $DATADIR/kpf/L1 = {DEFAULT_L1_DIR})",
    )
    p.add_argument(
        "--csv-dir", default=DEFAULT_CSV_DIR,
        help=f"base dir for SoCal RV CSVs "
             f"(default: $BASEDIR/solar_csvs = {DEFAULT_CSV_DIR})",
    )
    p.add_argument("--csv", default=None, help="explicit path to the SoCal RV CSV")
    p.add_argument(
        "--glob", action="store_true",
        help="discover L1 files by globbing instead of reading the RV CSV",
    )
    p.add_argument(
        "--files", nargs="+", default=None,
        help="explicit list of L1 files (overrides --glob/--csv)",
    )
    p.add_argument(
        "--all-sky", action="store_true",
        help="do not restrict to clear-sky spectra from the CSV",
    )

    # Processing options
    p.add_argument(
        "--norm-percentile", type=float, default=95.0,
        help="per-order normalization percentile (default: 95)",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="skip per-order normalization",
    )
    p.add_argument(
        "--n-modes", type=int, default=None,
        help="number of Fourier modes for --method frizzle "
             "(default: ~npix/3, odd)",
    )
    p.add_argument(
        "--nproc", type=int, default=1,
        help="number of processes to parallelize the orders over; each order "
             "is resampled independently, a batch of nproc at a time "
             "(default: 1 = serial)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="process at most this many spectra (for quick tests)",
    )

    # Output
    p.add_argument(
        "--outdir", default="outputs",
        help="directory to write the output .npz into, created if missing; "
             "ignored when --output is itself a path with a directory "
             "(default: outputs)",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="output .npz filename or path "
             "(default: socal_spectra_{date}_{trace}_{method}_common_wavegrid.npz "
             "inside --outdir)",
    )
    p.add_argument(
        "--save-errors", action="store_true",
        help="also store the resampled flux errors under key 'eflux'",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="less output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet

    l1files = resolve_l1_files(args)
    if args.limit is not None:
        l1files = l1files[: args.limit]
    print(f"Reading {len(l1files)} L1 files for {args.date} ...")

    data = read_l1_spectra(l1files, getblaze=False, verbose=verbose)
    nspec, ntrace, norder, npix = data["fluxs"].shape
    print(f"Loaded {nspec} spectra x {ntrace} traces x {norder} orders x {npix} pix")

    # Shared pre-step: normalize each order.
    if not args.no_normalize:
        data["fluxs"], data["vars"] = normalize_orders(
            data["fluxs"], data["vars"], percentile=args.norm_percentile
        )

    if args.nproc > 1:
        print(f"Parallelizing orders over {args.nproc} processes")

    if args.trace == "ALL":
        # Common grid = SCI1 wavelengths of the first epoch (as in the notebook).
        common_grid = data["waves"][0, TRACE_INDEX["SCI1"]]

        if args.combine == "frizzle":
            # Forward-model all three traces together with frizzle (one call per
            # epoch/order); --method is not used for the combination.
            print(f"Trace: ALL   Combine: frizzle-stack")
            flux, err = frizzle_stack_trace_all_orders(
                data, common_grid, n_modes=args.n_modes,
                nproc=args.nproc, verbose=verbose,
            )
        else:
            # Resample each trace with --method onto the common grid, then
            # flux-weight sum them per epoch.
            print(f"Method: {args.method}   Trace: ALL   Combine: sum")
            flux_per_trace, err_per_trace = [], []
            for name in ("SCI1", "SCI2", "SCI3"):
                t = TRACE_INDEX[name]
                print(f"[{name}]")
                fo, eo = shift_and_resample_trace(
                    data, t, args.method, common_grid, n_modes=args.n_modes,
                    nproc=args.nproc, verbose=verbose,
                )
                flux_per_trace.append(fo)
                err_per_trace.append(eo)

            flux = np.empty((nspec, norder, npix))
            err = np.empty((nspec, norder, npix))
            for j in range(nspec):
                f_traces = [flux_per_trace[t][j] for t in range(NTRACE)]
                e_traces = [err_per_trace[t][j] for t in range(NTRACE)]
                flux[j], err[j] = combine_traces(f_traces, e_traces)
    else:
        print(f"Method: {args.method}   Trace: {args.trace}")
        t = TRACE_INDEX[args.trace]
        common_grid = data["waves"][0, t]
        flux, err = shift_and_resample_trace(
            data, t, args.method, common_grid, n_modes=args.n_modes,
            nproc=args.nproc, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Save, matching the notebook's npz layout.
    # ------------------------------------------------------------------
    # Tag the output with the resampling scheme actually used: for ALL with
    # frizzle-stacking, --method is irrelevant, so name it "frizzlestack".
    scheme = (
        "frizzlestack"
        if (args.trace == "ALL" and args.combine == "frizzle")
        else args.method
    )
    output = args.output or (
        f"socal_spectra_{args.date}_{args.trace}_{scheme}"
        "_common_wavegrid.npz"
    )
    # Place the file in --outdir unless the user gave --output with its own
    # directory. Create whatever directory we end up writing into.
    if os.path.dirname(output) == "":
        output = os.path.join(args.outdir, output)
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out = dict(
        wave=common_grid,
        flux=flux,
        bjd=data["bjds"],
        texp=data["texps"],
        filename=data["files"],
    )
    if args.save_errors:
        out["eflux"] = err
    np.savez(output, **out)
    print(f"Wrote {output}")
    print(f"  wave {out['wave'].shape}, flux {out['flux'].shape}, "
          f"bjd {out['bjd'].shape}, texp {out['texp'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
