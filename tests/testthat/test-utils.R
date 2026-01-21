# Tests for utility functions
#
# These tests verify internal utility functions

# =============================================================================
# gpu_type_from_r() Tests
# =============================================================================

test_that("gpu_type_from_r() handles logical", {
  expect_equal(gpu_type_from_r(c(TRUE, FALSE)), "BOOL8")
})

test_that("gpu_type_from_r() handles integer", {
  expect_equal(gpu_type_from_r(1:10), "INT32")
})

test_that("gpu_type_from_r() handles double", {
  expect_equal(gpu_type_from_r(c(1.5, 2.5)), "FLOAT64")
})

test_that("gpu_type_from_r() handles character", {
  expect_equal(gpu_type_from_r(c("a", "b")), "STRING")
})

test_that("gpu_type_from_r() handles Date", {
  expect_equal(gpu_type_from_r(as.Date("2024-01-01")), "TIMESTAMP_DAYS")
})

test_that("gpu_type_from_r() handles POSIXct", {
  expect_equal(gpu_type_from_r(as.POSIXct("2024-01-01")), "TIMESTAMP_MICROSECONDS")
})

test_that("gpu_type_from_r() handles factor", {
  expect_equal(gpu_type_from_r(factor(c("a", "b"))), "DICTIONARY32")
})

# =============================================================================
# col_index() Tests
# =============================================================================

test_that("col_index() returns 0-based index", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  # First column (mpg) should be index 0
  expect_equal(col_index(gpu_df, "mpg"), 0L)

  # Second column (cyl) should be index 1
  expect_equal(col_index(gpu_df, "cyl"), 1L)

  # Last column (carb) should be index 10
  expect_equal(col_index(gpu_df, "carb"), 10L)
})

test_that("col_index() errors on non-existent column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)

  expect_error(col_index(gpu_df, "nonexistent"), "not found")
})

test_that("col_index() works after select", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(mtcars)
  selected <- dplyr::select(gpu_df, hp, mpg, wt)

  # hp is now first (index 0)
  expect_equal(col_index(selected, "hp"), 0L)

  # mpg is now second (index 1)
  expect_equal(col_index(selected, "mpg"), 1L)

  # wt is now third (index 2)
  expect_equal(col_index(selected, "wt"), 2L)

  # Old columns should error
  expect_error(col_index(selected, "cyl"), "not found")
})
