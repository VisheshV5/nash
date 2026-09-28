#include <pybind11/pybind11.h>

#include "regret/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Regret native core (engine, abstraction, CFR, search).";
  m.def("version", [] { return regret::kVersion; }, "Version string the extension was built with.");
}
