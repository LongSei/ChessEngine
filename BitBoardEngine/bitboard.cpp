
#include "bitboard.h"



PYBIND11_MODULE(BitBoardEngine, h)
{
    h.doc() = "This is some function for bitboardengine, this is doc for the module";
    h.def("some_fn", &some_fn);
}