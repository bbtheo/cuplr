#ifndef CUPLYR_OPS_COMMON_HPP
#define CUPLYR_OPS_COMMON_HPP

#include <Rcpp.h>
#include <cudf/binaryop.hpp>

#include <string>

namespace cuplyr {

inline cudf::binary_operator get_compare_op(const std::string& op) {
    if (op == "==") return cudf::binary_operator::EQUAL;
    if (op == "!=") return cudf::binary_operator::NOT_EQUAL;
    if (op == ">")  return cudf::binary_operator::GREATER;
    if (op == ">=") return cudf::binary_operator::GREATER_EQUAL;
    if (op == "<")  return cudf::binary_operator::LESS;
    if (op == "<=") return cudf::binary_operator::LESS_EQUAL;
    Rcpp::stop("Unknown comparison operator: %s", op.c_str());
}

inline cudf::binary_operator get_arith_op(const std::string& op) {
    if (op == "+") return cudf::binary_operator::ADD;
    if (op == "-") return cudf::binary_operator::SUB;
    if (op == "*") return cudf::binary_operator::MUL;
    if (op == "/") return cudf::binary_operator::TRUE_DIV;
    if (op == "%/%") return cudf::binary_operator::FLOOR_DIV;
    if (op == "%%") return cudf::binary_operator::MOD;
    if (op == "^") return cudf::binary_operator::POW;
    Rcpp::stop("Unknown arithmetic operator: %s", op.c_str());
}

} // namespace cuplyr

#endif // CUPLYR_OPS_COMMON_HPP
