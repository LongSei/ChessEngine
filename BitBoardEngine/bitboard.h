
#ifndef BITBOARD_H
#define BITBOARD_H
#include "pybind11/pybind11.h"
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
     * @enum enumFile
     * @brief Bitmask representation of chess files
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

    // Use enum classes for type safety
    enum class PieceColor : uint8_t
    {
        ePC_WHITE,
        ePC_BLACK
    };
    enum class PieceType : uint8_t
    {
        ePT_PAWN,
        ePT_KNIGHT,
        ePT_BISHOP,
        ePT_ROOK,
        ePT_QUEEN,
        ePT_KING,
        ePT_NONE
    };

    enum enumMoveType
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
     * Encoding format:
     * - Bits 0-5:   Origin square (0-63)
     * - Bits 6-11:  Destination square (0-63)
     * - Bits 12-15: Move flags (enumMoveType)
     * - Bits 16-17: Promotion piece type (if applicable)
     */
    class Move
    {
    private:
        uint16_t data;

    public:
        /**
         * @brief Construct a new Move object
         * @param from Square index (0-63)
         * @param to Square index (0-63)
         * @param flags Move type flags (enumMoveType)
         * @param promotion Promotion piece type (if applicable)
         */
        Move(uint8_t from, uint8_t to, uint8_t flags, uint8_t promotion = 0);

        /// @brief Get origin square index
        uint8_t from() const;
        /// @brief Get destination square index
        uint8_t to() const;
        /// @brief Get flags
        uint8_t flags() const;
    };
    bool is_move_legal(const Move &move) const;
    bool is_pseudo_legal(const Move &move) const;

    /**
     * @brief Scored Move for alpha-beta search
     *
     */
    struct ScoredMove
    {
        Move move;
        int score; // Heuristic score for sorting
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
        BitBoard getPieceBitBoard(enumPieceColor color, enumPieceType type) const;
        BitBoard getWhiteOccupancy() const;
        BitBoard getBlackOccupancy() const;
        BitBoard getAllOccupancy() const;
        enumPieceType getPieceType(BitBoard square) const; // ChessBoard
        void set_piece(enumPieceColor color, enumPieceType type, BitBoard squares);
        void clear_piece(enumPieceColor color, enumPieceType type, BitBoard squares);
        void reset_to_initial_position(); // ChessBoard
        std::string to_unicode() const;   // Visual board representation
        void print_debug() const;         // ASCII KQkq white/black output
    };

    struct StateDelta
    {
        Move move;
        BitBoard captured_piece;
        uint8_t prev_castling;
        BitBoard prev_en_passant;
        int prev_halfmove;
    };

    /**
     * @struct GameState
     * @brief Complete game state representation
     *
     * Contains board position and game rules metadata.
     */
    struct GameState
    {
        std::vector<StateDelta> undo_stack;
        ChessBoard board;           ///< Current board position
        bool is_white_turn;         ///< Side to move
        BitBoard en_passant_square; ///< En passant target square
        uint8_t castling_rights;    ///< Castling availability (4 bits: KQkq)
        int half_move_clock;        ///< 50-move rule counter
        int full_move_number;       ///< Game move counter

        /// @brief Generate all legal moves for current position
        std::vector<Move> generate_legal_moves();

        /**
         * @brief Check if the current player is in check
         *
         * @param state
         * @return true
         * @return false
         */
        bool isInCheck();

        void makeMove(const Move &move);

        /**
         * @brief for optimize memory in searching
         *
         * @param move move to undo
         */
        void unmakeMove(const Move &move); // GameState
        bool isCheckmate() const;
        bool isStalemate() const;
        bool isDraw() const; // 50-move rule/threefold repetition
        std::string toFen() const;
        void fromFen(const std::string &fen);
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