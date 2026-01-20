#' Create a GPU-backed tibble
#'
#' @param data A data frame to transfer to GPU memory
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object
#' @export
#' @examples
#' if (interactive()) {
#'   df <- data.frame(x = 1:1000, y = rnorm(1000))
#'   gpu_df <- tbl_gpu(df)
#'   gpu_df
#' }
tbl_gpu <- function(data, ...) {
  UseMethod("tbl_gpu")
}

#' @export
tbl_gpu.data.frame <- function(data, ...) {
  # Transfer to GPU
  ptr <- .Call(`_cuplr_df_to_gpu`, data)

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  new_tbl_gpu(ptr = ptr, schema = schema)
}

#' @export
tbl_gpu.tbl_gpu <- function(data, ...) {
  data
}

# Internal constructor
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = list(),
                        groups = character()) {
  structure(
    list(
      ptr = ptr,
      schema = schema,
      lazy_ops = lazy_ops,
      groups = groups
    ),
    class = c("tbl_gpu", "tbl_lazy", "tbl")
  )
}

#' @export
is_tbl_gpu <- function(x) {
  inherits(x, "tbl_gpu")
}

#' @export
as_tbl_gpu <- function(x, ...) {
  tbl_gpu(x, ...)
}

# Print method - glimpse-style output
#' @export
print.tbl_gpu <- function(x, ..., width = NULL) {
  width <- width %||% getOption("width", 80)

  if (is.null(x$ptr)) {
    cat("# A GPU tibble [lazy, not materialized]\n")
    cat("# Schema: ", paste(x$schema$names, collapse = ", "), "\n")
    cat("# Operations pending: ", length(x$lazy_ops), "\n")
    return(invisible(x))
  }

  dims <- dim(x)
  types <- gpu_col_types(x$ptr)
  col_names <- x$schema$names

  # Header
  cat("Rows: ", format(dims[1], big.mark = ","), "\n", sep = "")
  cat("Columns: ", dims[2], "\n", sep = "")

  if (length(x$groups) > 0) {
    cat("Groups: ", paste(x$groups, collapse = ", "), "\n")
  }

  # Get preview data (first 10 rows for value display)
  preview <- tryCatch(
    gpu_head(x$ptr, 10L, col_names),
    error = function(e) NULL
  )

  # Calculate column name width for alignment
  max_name_width <- max(nchar(col_names), 1)

  # Print each column
  for (i in seq_along(col_names)) {
    col_name <- col_names[i]
    col_type <- types[i]

    # Format: $ name <type> value1, value2, ...
    name_pad <- format(col_name, width = max_name_width)
    type_str <- paste0("<", col_type, ">")

    # Get preview values
    if (!is.null(preview)) {
      vals <- preview[[i]]
      if (length(vals) > 0) {
        # Truncate strings for display
        if (is.character(vals)) {
          vals <- ifelse(nchar(vals) > 20, paste0(substr(vals, 1, 17), "..."), vals)
          vals <- paste0("\"", vals, "\"")
        }
        vals[is.na(vals)] <- "NA"
        val_str <- paste(vals, collapse = ", ")

        # Truncate to fit width
        available_width <- width - max_name_width - nchar(type_str) - 6
        if (nchar(val_str) > available_width && available_width > 10) {
          val_str <- paste0(substr(val_str, 1, available_width - 3), "...")
        }
      } else {
        val_str <- ""
      }
    } else {
      val_str <- "[preview unavailable]"
    }

    cat("$ ", name_pad, " ", type_str, " ", val_str, "\n", sep = "")
  }

  invisible(x)
}

#' @export
dim.tbl_gpu <- function(x) {
  if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    gpu_dim(x$ptr)
  }
}

#' @export
names.tbl_gpu <- function(x) {
  x$schema$names
}

#' @export
`names<-.tbl_gpu` <- function(x, value) {
  x$schema$names <- value
  x
}

# Type helper
gpu_type_from_r <- function(x) {
  if (is.logical(x)) return("BOOL8")
  if (is.integer(x)) return("INT32")
  if (is.double(x)) {
    if (inherits(x, "Date")) return("TIMESTAMP_DAYS")
    if (inherits(x, "POSIXct")) return("TIMESTAMP_MICROSECONDS")
    return("FLOAT64")
  }
  if (is.character(x)) return("STRING")
  if (is.factor(x)) return("DICTIONARY32")
  if (inherits(x, "integer64")) return("INT64")
  "UNKNOWN"
}
