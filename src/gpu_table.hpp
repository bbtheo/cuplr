// src/gpu_table.hpp
#ifndef CUPLYR_GPU_TABLE_HPP
#define CUPLYR_GPU_TABLE_HPP

#include <Rcpp.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>

namespace cuplyr {

// Wrap cudf::table in a shared_ptr for R interop
using GpuTablePtr = std::shared_ptr<cudf::table>;

// Custom destructor that ensures GPU cleanup
inline void release_gpu_table(GpuTablePtr* ptr) {
    if (ptr != nullptr) {
        // Reset triggers cudf::table destructor, freeing GPU memory
        ptr->reset();
        delete ptr;
    }
}

// Create XPtr with custom destructor
inline Rcpp::XPtr<GpuTablePtr> make_gpu_table_xptr(std::unique_ptr<cudf::table> tbl) {
    // Convert unique_ptr to shared_ptr for R ownership
    auto* sptr = new GpuTablePtr(std::move(tbl));
    return Rcpp::XPtr<GpuTablePtr>(sptr, true);  // true = register destructor
}

// Extract table_view from XPtr (non-owning view)
inline cudf::table_view get_table_view(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return (*xptr)->view();
}

// Get mutable table reference
inline cudf::table& get_table_ref(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return **xptr;
}

} // namespace cuplyr

#endif // CUPLYR_GPU_TABLE_HPP