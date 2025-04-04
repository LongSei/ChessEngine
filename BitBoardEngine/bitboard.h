
#ifndef BITBOARD_H
#define BITBOARD_H
#include "pybind11/pybind11.h"
using BitBoard = uint64_t;
namespace BitBoardEngine
{

    /**
     * @brief 64 bit mask for Rank, every bit on the rank is 1, all other is 0
     *
     */
    enum enumRank
    {
        eRANK_1 = 0xFFULL << 0,
        eRANK_2 = 0xFFULL << 8,
        eRANK_3 = 0xFFULL << 16,
        eRANK_4 = 0xFFULL << 24,
        eRANK_5 = 0xFFULL << 32,
        eRANK_6 = 0xFFULL << 40,
        eRANK_7 = 0xFFULL << 48,
        eRANK_8 = 0xFFULL << 56,
    };

    /**
     * @brief 64 bit mask for File, every bit on the file is 1, all other is 0
     *
     */
    enum enumFile
    {
        eFILE_A = 0x0101010101010101,
        eFILE_B = 0x0202020202020202,
        eFILE_C = 0x0404040404040404,
        eFILE_D = 0x0808080808080808,
        eFILE_E = 0x1010101010101010,
        eFILE_F = 0x2020202020202020,
        eFILE_G = 0x4040404040404040,
        eFILE_H = 0x8080808080808080
    };

    enum enumPieceColor
    {
        ePC_WHITE = 0,
        ePC_BLACK = 1
    };

    enum enumPieceType
    {
        ePT_NONE = -1,
        ePT_PAWN = 0,
        ePT_KNIGHT = 1,
        ePT_BISHOP = 2,
        ePT_ROOK = 3,
        ePT_QUEEN = 4,
        ePT_KING = 5
    };

    /**
     * @brief ChessBoard is a class only contain information about the state of the board only, not the state of the game
     *
     */
    struct ChessBoard
    {
        BitBoard pieces[2][6];

        ChessBoard();
        BitBoard getPieceBitBoard(enumPieceColor color, enumPieceType type) const;
        BitBoard getWhiteOccupancy() const;
        BitBoard getBlackOccupancy() const;
        BitBoard getAllOccupancy() const;
        enumPieceType getPieceType(BitBoard square) const;
    };

    /**
     * @brief Initialize the attacks table and other necessary stuff
     *
     */
    void init();

    void initKnightAttacksTable();
    void initKingAttacksTable();
    void initPawnAttacksTable();

    
}

#endif