#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
bool FastUniqueC(SEXP x) {
  switch (TYPEOF(x)) {
  case INTSXP: { // Integer vectors
    IntegerVector vec(x);
    int first_value = vec[0];
    for (int i = 1; i < vec.size(); ++i) {
      if (vec[i] != first_value) {
        return false;
      }
    }
    break;
  }
  case REALSXP: { // Numeric vectors
    NumericVector vec(x);
    double first_value = vec[0];
    for (int i = 1; i < vec.size(); ++i) {
      if (vec[i] != first_value) {
        return false;
      }
    }
    break;
  }
  case STRSXP: { // Character vectors
    CharacterVector vec(x);
    String first_value = vec[0];
    for (int i = 1; i < vec.size(); ++i) {
      if (vec[i] != first_value) {
        return false;
      }
    }
    break;
  }
  default: {
    stop("Unsupported vector type.");
  }
  }
  return true;
}
