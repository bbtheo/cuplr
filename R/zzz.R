#' @useDynLib cuplr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom rlang %||%
NULL

.onLoad <- function(libname, pkgname) {
  # Check GPU availability
  gpu_ok <- TRUE
  # Set package options
  op <- options()
  op.cuplr <- list(
    cuplr.verbose = FALSE,
    cuplr.lazy = TRUE,
    cuplr.gpu_available = gpu_ok
  )
  toset <- !(names(op.cuplr) %in% names(op))
  if (any(toset)) options(op.cuplr[toset])

  invisible()
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("cuplr: GPU-accelerated data manipulation loaded")
  # TODO: implement gpu_info() to show GPU details
}
