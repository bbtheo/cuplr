# cuplr – Notes for Coding Agents

This package implements a GPU-backed dplyr-like API in R, with C++/Rcpp bindings to libcudf. Most failures are type-mapping, build environment, or memory/allocation semantics issues.

## Architecture Snapshot
- `R/` implements dplyr verbs and schema tracking.
- `src/` implements GPU ops; objects are `cudf::table` wrapped in `Rcpp::XPtr`.
- `gpu_table.hpp` owns lifetime; rely on it for pointer safety.

## Local Dev (pixi)
Use the built-in pixi tasks for day-to-day iteration:
- `pixi run configure` -> generates `src/Makevars`.
- `pixi run install` -> builds and installs the package.
- `pixi run load-dev` -> quick reload in an R session without reinstall.
- `pixi run test` -> testthat (GPU required).

Common failure: missing cudf headers. In this environment, `bitmask_allocation_size_bytes` is in `cudf/null_mask.hpp` (not `cudf/bitmask.hpp`).

## Known Tricky Areas

### 1) Type Mapping Consistency
R schema types must match actual GPU column types.
- `R/utils.R::gpu_type_from_r()`
- `src/transfer_io.cpp::df_to_gpu()`
If you change one, update the other and tests.

### 2) Logical/Date/POSIXct Handling
Logicals are stored as BOOL8; Date uses TIMESTAMP_DAYS; POSIXct uses TIMESTAMP_MICROSECONDS.
Collect/head must return R classes (`Date`, `POSIXct`) not raw numbers.

### 3) Rcpp Exports
Moving C++ functions requires regenerating exports:
- Run `Rcpp::compileAttributes()` (or `devtools::document()`).

### 4) Memory Semantics
Most ops allocate new tables. Replacement mutates were optimized to avoid append+select, but you still need to consider GC and GPU memory pressure for large pipelines.

### 5) INT64 Precision
`gpu_collect()` returns `NumericVector` for INT64. A warning is raised when values exceed 2^53. If you change INT64 handling, update the warning behavior and tests.

## Source Map (post-split)
- `src/transfer_io.cpp`: conversion, collect/head, df_to_gpu.
- `src/ops_filter.cpp`: filter ops.
- `src/ops_compare.cpp`: compare ops for summarise temp cols.
- `src/ops_mutate.cpp`: mutate and copy helpers.
- `src/ops_select.cpp`: select ops.
- `src/ops_groupby.cpp`: groupby/summarise.
- `src/gpu_info.cpp`: GPU availability/info.
- `src/cuda_utils.hpp`: CUDA error helper.
- `src/ops_common.hpp`: shared operator mapping.

## Adding Features (local loop)
1) Implement C++ in the right `src/ops_*.cpp`.
2) Add R wrapper in `R/`.
3) Regenerate exports.
4) Add tests in `tests/testthat/`.

## Minimal Local Sanity Checks
- `pixi run install`
- `pixi run test` (requires GPU)

If no GPU is available, rely on tests with `skip_if_no_gpu()` and validate logic by inspection or with small CPU-side checks.
