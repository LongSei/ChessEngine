
#include "bitboard.h"

namespace BitBoardEngine
{

    void initNonSliderAttacks()
    {

    }
    void initSliderAttacks()
    {
    }
    void init()
    {
        initNonSliderAttacks();
        initSliderAttacks();
    }
}

PYBIND11_MODULE(BitBoardEngine, h)
{
    h.doc() = "This is some function for bitboardengine, this is doc for the module";
    h.def("init", &BitBoardEngine::init, "Init core data for engine to work, please call this before any use of the engine");
    
    
    py::class_<BitBoardEngine::Move>(h, "Move")
        .def(py::init<uint8_t,uint8_t,BitBoardEngine::MoveType,BitBoardEngine::PieceType>());

}