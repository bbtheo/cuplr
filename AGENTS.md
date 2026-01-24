# cuplr Agent Notes

This repo mixes R and C++ (Rcpp) for GPU-backed dplyr-like operations using libcudf. The trickiest parts are build tooling, GPU availability, and keeping R-level schemas aligned with GPU types.

## Local Dev (pixi)
Default workflow when iterating locally:
- `pixi run configure` generates `src/Makevars` pointing at CUDA/cudf/RMM.
- `pixi run install` builds + installs the R package.
- `pixi run load-dev` loads the package for quick checks without reinstall.
- `pixi run test` runs testthat (GPU required).
- `pixi run dev` cleans + rebuilds (slow but consistent).

## Current C++ Layout (split from transfer.cpp)
- `src/transfer_io.cpp`: R <-> GPU conversion, collect/head/df_to_gpu.
- `src/ops_filter.cpp`: filter operations and mask handling.
- `src/ops_compare.cpp`: comparison ops used for summarise temp columns.
- `src/ops_mutate.cpp`: mutate and copy/replace ops.
- `src/ops_select.cpp`: select operations.
- `src/ops_groupby.cpp`: summarise/groupby logic.
- `src/gpu_info.cpp`: device availability/info.
- `src/cuda_utils.hpp`: `check_cuda()` helper.
- `src/ops_common.hpp`: shared operator mapping helpers.
- `src/gpu_table.hpp`: pointer ownership helpers for cudf::table.

## R Entry Points
- `R/tbl-gpu.R`: `tbl_gpu()` constructor + schema metadata.
- `R/mutate.R`, `R/filter.R`, `R/select.R`, `R/summarise.R`: dplyr verbs.
- `R/collect.R`: pulls data back to R and warns on INT64 precision.
- `R/gpu-memory.R`: memory reporting and GC helpers.

## Known Sharp Edges (things that were hard)
- **cudf header names differ by version.** `bitmask_allocation_size_bytes` lives in `cudf/null_mask.hpp` in this environment. Avoid `cudf/bitmask.hpp`.
- **R type vs GPU type mismatch** can silently break results. Keep `gpu_type_from_r()` and `df_to_gpu()` in sync (logical/Date/POSIXct especially).
- **GPU not detected** is common in CI or local dev. Tests use `skip_if_no_gpu()`; don’t remove it.
- **Rcpp exports need regeneration** after moving/adding functions: run `Rcpp::compileAttributes()` or `devtools::document()`.
- **INT64 precision**: `gpu_collect()` returns doubles; warn when values exceed 2^53.
- **Memory growth**: each GPU op tends to allocate new tables. Replacement mutate paths are optimized, but GC still matters.

## Debugging Local Build Failures
- If a cudf header can’t be found, check `pixi` environment paths and `src/Makevars`.
- Run `pixi run configure` after updating CUDA/cudf libs.
- Use `rg --files -g '*bitmask*' $CONDA_PREFIX/include/cudf` to locate moved headers.

## Adding New GPU Ops (local loop)
1) Implement in appropriate `src/ops_*.cpp` file.
2) Add an `// [[Rcpp::export]]` function.
3) Run `Rcpp::compileAttributes()` to regenerate `R/RcppExports.R` + `src/RcppExports.cpp`.
4) Add R wrapper in `R/*.R` and tests in `tests/testthat/`.

## When You Touch Types (local loop)
- Update schema: `R/utils.R` (`gpu_type_from_r`).
- Update collect/head conversion for new cudf types.
- Add tests for round-trip behavior.
