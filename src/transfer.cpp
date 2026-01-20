// src/transfer.cpp
#include "gpu_table.hpp"
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplr {

// Create GPU column from R numeric vector
std::unique_ptr<column> numeric_to_gpu(NumericVector x) {
    size_type n = x.size();

    // Allocate device memory
    rmm::device_buffer data(n * sizeof(double),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    // Copy from host to device
    cudaMemcpy(data.data(), &x[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Handle NAs by creating validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    // Check for NAs
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            // Clear bit i
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::FLOAT64},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R integer vector
std::unique_ptr<column> integer_to_gpu(IntegerVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    cudaMemcpy(data.data(), &x[0], n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Handle NAs
    rmm::device_buffer null_mask;
    size_type null_count = 0;
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);

    for (size_type i = 0; i < n; ++i) {
        if (IntegerVector::is_na(x[i])) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::INT32},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R character vector
std::unique_ptr<column> character_to_gpu(CharacterVector x) {
    size_type n = x.size();

    std::vector<std::string> strings(n);
    std::vector<bool> valids(n, true);

    for (size_type i = 0; i < n; ++i) {
        if (CharacterVector::is_na(x[i])) {
            valids[i] = false;
            strings[i] = "";
        } else {
            strings[i] = as<std::string>(x[i]);
        }
    }

    // Concatenate strings and create offsets
    std::string concatenated;
    std::vector<int32_t> offsets;
    offsets.push_back(0);

    for (const auto& s : strings) {
        concatenated += s;
        offsets.push_back(concatenated.size());
    }

    // Copy data to device
    rmm::device_buffer data(concatenated.size(),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource());
    cudaMemcpy(data.data(), concatenated.data(), concatenated.size(), cudaMemcpyHostToDevice);

    // Copy offsets to device
    rmm::device_buffer offsets_buf(offsets.data(), offsets.size() * sizeof(int32_t),
                                   rmm::cuda_stream_view(),
                                   rmm::mr::get_current_device_resource());

    // Create offsets column
    auto offsets_col = std::make_unique<column>(
        data_type{type_id::INT32},
        offsets.size(),
        std::move(offsets_buf),
        rmm::device_buffer{},
        0
    );

    // Create validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (!valids[i]) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource());
    }

    // Create STRING column with offsets as child
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(offsets_col));

    return std::make_unique<column>(
        data_type{type_id::STRING},
        n,
        std::move(data),
        std::move(null_mask),
        null_count,
        std::move(children)
    );
}

// Copy numeric column from GPU to R
NumericVector gpu_to_numeric(const cudf::column_view& col) {
    size_type n = col.size();
    NumericVector result(n);

    // Copy data from device to host
    cudaMemcpy(&result[0], col.data<double>(), n * sizeof(double), cudaMemcpyDeviceToHost);

    // Handle nulls
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);

        for (size_type i = 0; i < n; ++i) {
            if (!(validity[i / 8] & (1 << (i % 8)))) {
                result[i] = NA_REAL;
            }
        }
    }

    return result;
}

// Copy integer column from GPU to R
IntegerVector gpu_to_integer(const cudf::column_view& col) {
    size_type n = col.size();
    IntegerVector result(n);

    cudaMemcpy(&result[0], col.data<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);

        for (size_type i = 0; i < n; ++i) {
            if (!(validity[i / 8] & (1 << (i % 8)))) {
                result[i] = NA_INTEGER;
            }
        }
    }

    return result;
}

// Copy string column from GPU to R (simplified - assumes contiguous chars)
CharacterVector gpu_to_character(const cudf::column_view& col) {
    size_type n = col.size();
    CharacterVector result(n);

    // For strings, we need offsets and char data
    // This is a simplified implementation
    if (n == 0) return result;

    // Get offsets child column
    auto offsets_col = col.child(0);
    std::vector<int32_t> offsets(offsets_col.size());
    cudaMemcpy(offsets.data(), offsets_col.data<int32_t>(),
               offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Get char data
    size_t total_chars = offsets.back();
    std::vector<char> chars(total_chars);
    if (total_chars > 0) {
        cudaMemcpy(chars.data(), col.data<char>(), total_chars, cudaMemcpyDeviceToHost);
    }

    // Handle nulls
    std::vector<uint8_t> validity;
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        validity.resize(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);
    }

    // Build R strings
    for (size_type i = 0; i < n; ++i) {
        if (!validity.empty() && !(validity[i / 8] & (1 << (i % 8)))) {
            result[i] = NA_STRING;
        } else {
            int32_t start = offsets[i];
            int32_t len = offsets[i + 1] - start;
            result[i] = std::string(chars.data() + start, len);
        }
    }

    return result;
}

} // namespace cuplr

// [[Rcpp::export]]
IntegerVector gpu_dim(SEXP xptr) {
    using namespace cuplr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    return IntegerVector::create(view.num_rows(), view.num_columns());
}

// [[Rcpp::export]]
List gpu_head(SEXP xptr, int n, CharacterVector col_names) {
    using namespace cuplr;
    using namespace cudf;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int nrow = std::min(static_cast<int>(view.num_rows()), n);
    int ncol = view.num_columns();

    List result(ncol);

    for (int i = 0; i < ncol; ++i) {
        column_view col = view.column(i);

        // Create a view of just the first n rows
        column_view head_col(col.type(), nrow, col.head(), col.null_mask(), col.null_count(), col.offset());

        switch (col.type().id()) {
            case type_id::FLOAT64:
                result[i] = gpu_to_numeric(head_col);
                break;
            case type_id::FLOAT32: {
                // Convert float to double for R
                std::vector<float> temp(nrow);
                cudaMemcpy(temp.data(), head_col.data<float>(), nrow * sizeof(float), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = temp[j];
                result[i] = rv;
                break;
            }
            case type_id::INT32:
            case type_id::BOOL8:
                result[i] = gpu_to_integer(head_col);
                break;
            case type_id::INT64: {
                // Convert int64 to double for R (may lose precision)
                std::vector<int64_t> temp(nrow);
                cudaMemcpy(temp.data(), head_col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                result[i] = rv;
                break;
            }
            case type_id::STRING: {
                // For strings, we need to be more careful with head
                // Just collect all for now and truncate in R
                result[i] = gpu_to_character(col);
                break;
            }
            default:
                // Unsupported type - return NAs
                result[i] = NumericVector(nrow, NA_REAL);
        }
    }

    result.names() = col_names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);

    return result;
}

// [[Rcpp::export]]
CharacterVector gpu_col_types(SEXP xptr) {
    using namespace cuplr;
    using namespace cudf;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int ncol = view.num_columns();
    CharacterVector result(ncol);

    for (int i = 0; i < ncol; ++i) {
        switch (view.column(i).type().id()) {
            case type_id::FLOAT64: result[i] = "dbl"; break;
            case type_id::FLOAT32: result[i] = "flt"; break;
            case type_id::INT64: result[i] = "i64"; break;
            case type_id::INT32: result[i] = "int"; break;
            case type_id::INT16: result[i] = "i16"; break;
            case type_id::INT8: result[i] = "i8"; break;
            case type_id::BOOL8: result[i] = "lgl"; break;
            case type_id::STRING: result[i] = "chr"; break;
            case type_id::TIMESTAMP_DAYS: result[i] = "date"; break;
            case type_id::TIMESTAMP_SECONDS:
            case type_id::TIMESTAMP_MILLISECONDS:
            case type_id::TIMESTAMP_MICROSECONDS:
            case type_id::TIMESTAMP_NANOSECONDS:
                result[i] = "dttm"; break;
            default: result[i] = "???"; break;
        }
    }

    return result;
}

// [[Rcpp::export]]
SEXP df_to_gpu(DataFrame df) {
    using namespace cuplr;

    int ncol = df.size();
    CharacterVector names = df.names();

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        switch (TYPEOF(col)) {
            case REALSXP:
                columns.push_back(numeric_to_gpu(col));
                break;
            case INTSXP:
                columns.push_back(integer_to_gpu(col));
                break;
            case STRSXP:
                columns.push_back(character_to_gpu(col));
                break;
            case LGLSXP:
                // Convert logical to integer then to BOOL8
                columns.push_back(integer_to_gpu(as<IntegerVector>(col)));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(tbl));
}
