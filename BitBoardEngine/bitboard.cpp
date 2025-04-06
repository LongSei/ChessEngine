
#include "bitboard.h"



PYBIND11_MODULE(BitBoardEngine, h)
{
    h.doc() = "This is some function for bitboardengine, this is doc for the module";
    h.def("some_fn", &some_fn);
}
// Add pybind11 module definition
PYBIND11_MODULE(bitboard, m) {
    pybind11::enum_<enumPieceColor>(m, "PieceColor")
        .value("WHITE", ePC_WHITE)
        .value("BLACK", ePC_BLACK);
    // ... other enums and class bindings
}