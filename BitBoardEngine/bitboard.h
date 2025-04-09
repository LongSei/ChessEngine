
#ifndef BITBOARD_H
#define BITBOARD_H
#include "pybind11/pybind11.h"

namespace py = pybind11;

using BitBoard = uint64_t;

/**
 * @file bitboard.h
 * @brief Bitboard-based chess engine implementation
 *
 * This module implements a chess engine using bitboard representation.
 * All Python API bindings use snake_case naming convention.
 */
namespace BitBoardEngine
{

    /**
     * @enum enumRank
     * @brief Bitmask representation of chess ranks
     */
    enum class Rank : BitBoard
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
     * @enum enumFile
     * @brief Bitmask representation of chess files
     */
    enum class File : BitBoard
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

    // Use enum classes for type safety
    enum class PieceColor : uint8_t
    {
        ePC_WHITE = 0,
        ePC_BLACK = 1
    };
    enum class PieceType : uint8_t
    {
        ePT_PAWN = 0,
        ePT_KNIGHT = 1 << 0,
        ePT_BISHOP = 1 << 1,
        ePT_ROOK = 1 << 2,
        ePT_QUEEN = 1 << 3,
        ePT_KING = 1 << 4,
        ePT_NONE = 1 << 5
    };

    enum MoveType
    {
        eMT_QUIET = 0b0000,           // Regular move
        eMT_CAPTURE = 0b0001,         // Capture
        eMT_EN_PASSANT = 0b0010,      // En passant capture
        eMT_CASTLE_KS = 0b0011,       // Kingside castle
        eMT_CASTLE_QS = 0b0100,       // Queenside castle
        eMT_PROMOTION = 0b1000,       // Promotion (OR with piece type: KNIGHT=0, BISHOP=1, ROOK=2, QUEEN=3)
        eMT_DOUBLE_PAWN_PUSH = 0b0101 // For updating en passant
    };

    /**
     * @class Move
     * @brief Compact move representation using 32-bit storage
     *
     */
    class Move
    {
    private:
        uint8_t from;
        uint8_t to;
        MoveType flags;
        PieceType promotion;

    public:
        /**
         * @brief Construct a new Move object
         * @param from Square index (0-63)
         * @param to Square index (0-63)
         * @param flags Move type flags (MoveType)
         * @param promotion Promotion piece type (if applicable)
         */
        Move(uint8_t from, uint8_t to, MoveType flags, PieceType promotion = PieceType::ePT_NONE);

        /// @brief Get origin square index
        uint8_t getFrom() const;
        /// @brief Get destination square index
        uint8_t getTo() const;
        /// @brief Get flags
        MoveType getFlags() const;
        /// @brief Get promoted PieceType
        PieceType getPromotion() const;
    };

    /**
     * @struct ChessBoard
     * @brief Pure board state representation
     *
     * Contains only piece positions without game state metadata.
     * Uses 2x6 bitboards (white and black pieces by type).
     */
    struct ChessBoard
    {
        BitBoard pieces[2][6];

        ChessBoard();
        BitBoard getPieceBitBoard(PieceColor color, PieceType type) const;
        BitBoard getWhiteOccupancy() const;
        BitBoard getBlackOccupancy() const;
        BitBoard getAllOccupancy() const;
        std::vector<Move> generatePseudoLegalMoves() const;
        /// @brief Make move that change board representation of this instance
        void makeMoveInplace(const Move &move);
        /// @brief Return a board after the move from this instance
        ChessBoard makeMove(const Move &move);
        /// @brief Reset board state to default position
        void reset();
    };

    /**
     * @brief Initialize the attacks table and other necessary stuff
     *
     */
    void init();

    // ATTENTION: UNDERLYING CODE WILL NOT HAVE AN PYTHON API, THIS IS JUST FOR THE ENGINE TO WORK
    struct Magic
    {
        BitBoard mask;
        BitBoard magic;
        int shift;
    };

    Magic rookMagics[64];
    Magic bishopMagics[64];

    BitBoard rookAttacks[64][4096];
    BitBoard bishopAttacks[64][512];
    BitBoard knightAttacks[64];
    BitBoard kingAttacks[64];
    BitBoard queenAttacks[64];
    BitBoard pawnAttacks[64][2]; // square color
    void initSliderAttacks();
    void initNonSliderAttacks();

    // Just built-in popcount
    int count_bits(BitBoard b);
    // Generate blocker permutations
    BitBoard index_to_bitboard(int index, BitBoard mask);
    BitBoard calculate_rook_attacks(int sq, BitBoard blockers);
}

#endif