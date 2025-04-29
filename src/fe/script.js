// Mapping từ ký tự Unicode sang tên file ảnh SVG
const pieceImageMap = {
  "♜": "b_rook.svg",
  "♞": "b_knight.svg",
  "♝": "b_bishop.svg",
  "♛": "b_queen.svg",
  "♚": "b_king.svg",
  "♟": "b_pawn.svg",
  "♖": "w_rook.svg",
  "♘": "w_knight.svg",
  "♗": "w_bishop.svg",
  "♕": "w_queen.svg",
  "♔": "w_king.svg",
  "♙": "w_pawn.svg"
};

// Unicode chess pieces starting positions
const initialBoard = [
  ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
  ["♟", "♟", "♟", "♟", "♟", "♟", "♟", "♟"],
  Array(8).fill(""),
  Array(8).fill(""),
  Array(8).fill(""),
  Array(8).fill(""),
  ["♙", "♙", "♙", "♙", "♙", "♙", "♙", "♙"],
  ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"],
];

let selected = null;
let possibleMoves = [];
let currentPlayer = 'white'; // Thêm biến để theo dõi lượt chơi

// Utility: generate possible moves (basic logic)
function getPossibleMoves(r, c) {
  const piece = initialBoard[r][c];
  const moves = [];
  switch (piece) {
    case "♙": // white pawn
      if (r > 0 && !initialBoard[r - 1][c]) moves.push({row: r - 1, col: c, type: 'move'});
      // Thêm khả năng ăn chéo
      if (r > 0 && c > 0 && initialBoard[r - 1][c - 1] && isBlackPiece(initialBoard[r - 1][c - 1])) {
        moves.push({row: r - 1, col: c - 1, type: 'capture'});
      }
      if (r > 0 && c < 7 && initialBoard[r - 1][c + 1] && isBlackPiece(initialBoard[r - 1][c + 1])) {
        moves.push({row: r - 1, col: c + 1, type: 'capture'});
      }
      break;
    case "♟": // black pawn
      if (r < 7 && !initialBoard[r + 1][c]) moves.push({row: r + 1, col: c, type: 'move'});
      // Thêm khả năng ăn chéo
      if (r < 7 && c > 0 && initialBoard[r + 1][c - 1] && isWhitePiece(initialBoard[r + 1][c - 1])) {
        moves.push({row: r + 1, col: c - 1, type: 'capture'});
      }
      if (r < 7 && c < 7 && initialBoard[r + 1][c + 1] && isWhitePiece(initialBoard[r + 1][c + 1])) {
        moves.push({row: r + 1, col: c + 1, type: 'capture'});
      }
      break;
    case "♘":
    case "♞": // knight
      [
        [2, 1], [2, -1], [-2, 1], [-2, -1],
        [1, 2], [1, -2], [-1, 2], [-1, -2]
      ].forEach(([dr, dc]) => {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
          const target = initialBoard[nr][nc];
          if (!target || (piece === "♘" ? isBlackPiece(target) : isWhitePiece(target))) {
            moves.push({
              row: nr,
              col: nc,
              type: target ? 'capture' : 'move'
            });
          }
        }
      });
      break;
    // TODO: Thêm logic cho các quân cờ khác (rook, bishop, queen, king)
    default:
      break;
  }
  return moves;
}

// Hàm kiểm tra quân trắng
function isWhitePiece(piece) {
  return ['♔', '♕', '♖', '♗', '♘', '♙'].includes(piece);
}

// Hàm kiểm tra quân đen
function isBlackPiece(piece) {
  return ['♚', '♛', '♜', '♝', '♞', '♟'].includes(piece);
}


document.addEventListener("DOMContentLoaded", () => {
  const welcomeScreen = document.getElementById("welcome-screen");
  const gameScreen = document.getElementById("game-screen");
  const boardElement = document.getElementById("chessboard");
  const playAIBtn = document.getElementById("play-ai");
  const playHumanBtn = document.getElementById("play-human");
  const playerLabel = document.getElementById("player-label");
  const modeLabel = document.getElementById("mode-label");
  const turnLabel = document.getElementById("turn-label");
  const restartBtn = document.getElementById("restart-btn");
  const settingsBtn = document.getElementById("settings-btn");
  const modal = document.getElementById("settings-modal");
  const btnContinue = document.getElementById("modal-continue");
  const btnHome = document.getElementById("modal-home");

  function renderBoard() {
    boardElement.innerHTML = "";
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cell = document.createElement("div");
        if (r < 8 && c > 0) {
          cell.className = `square ${(r + c - 1) % 2 === 0 ? "light" : "dark"}`;
          cell.dataset.row = r;
          cell.dataset.col = c - 1;
          const p = initialBoard[r][c - 1];
          if (p) {
            const img = document.createElement("img");
            img.src = `images/${pieceImageMap[p]}`;
            img.className = "piece-img";
            img.alt = p;
            cell.appendChild(img);
          }
          cell.addEventListener("click", () => handleCellClick(r, c - 1));
          
          // Highlight ô được chọn
          if (selected && selected.row === r && selected.col === c - 1) {
            cell.classList.add("selected");
          }
          
          // Highlight các nước đi hợp lệ
          possibleMoves.forEach(move => {
            if (move.row === r && move.col === c - 1) {
              cell.classList.add(move.type === 'capture' ? 'capture' : 'possible-move');
            }
          });
        } else if (c === 0 && r < 8) {
          cell.className = "coord-label";
          cell.textContent = 8 - r;
        } else if (r === 8 && c > 0) {
          cell.className = "coord-label";
          cell.textContent = String.fromCharCode(64 + c);
        } else {
          cell.className = "coord-label";
        }
        boardElement.appendChild(cell);
      }
    }
  }

  function handleCellClick(row, col) {
    const piece = initialBoard[row][col];
    
    // Nếu đã có quân được chọn trước đó
    if (selected) {
      // Kiểm tra ô click có trong danh sách nước đi hợp lệ không
      const move = possibleMoves.find(m => m.row === row && m.col === col);
      
      if (move) {
        // Thực hiện di chuyển
        initialBoard[row][col] = initialBoard[selected.row][selected.col];
        initialBoard[selected.row][selected.col] = "";
        
        // Đổi lượt chơi
        currentPlayer = currentPlayer === 'white' ? 'black' : 'white';
        turnLabel.textContent = `Lượt: ${currentPlayer === 'white' ? 'Trắng' : 'Đen'}`;
      }
      
      resetSelection();
    } 
    // Nếu click vào quân cờ hợp lệ
    else if (piece && isValidPiece(piece, currentPlayer)) {
      selected = { row, col };
      possibleMoves = getPossibleMoves(row, col);
      highlightMoves();
    }
    
    renderBoard();
  }
  
  // Hàm highlight các ô có thể di chuyển
function highlightMoves() {
  // Xóa highlight cũ trước khi highlight mới
  document.querySelectorAll('.square').forEach(square => {
    square.classList.remove('selected', 'possible-move', 'capture');
  });

  // Highlight ô đang được chọn
  if (selected) {
    const selectedCell = document.querySelector(
      `[data-row="${selected.row}"][data-col="${selected.col}"]`
    );
    if (selectedCell) selectedCell.classList.add('selected');
  }

  // Highlight các ô có thể di chuyển
  possibleMoves.forEach(move => {
    const cell = document.querySelector(
      `[data-row="${move.row}"][data-col="${move.col}"]`
    );
    if (cell) {
      cell.classList.add(move.type === 'capture' ? 'capture' : 'possible-move');
    }
  });
}

// Hàm reset trạng thái chọn quân
function resetSelection() {
  selected = null;
  possibleMoves = [];
  highlightMoves(); // Cập nhật giao diện ngay lập tức
}

// Hàm kiểm tra quân cờ có thuộc người chơi hiện tại không
function isValidPiece(piece, player) {
  const whitePieces = ['♔', '♕', '♖', '♗', '♘', '♙'];
  const blackPieces = ['♚', '♛', '♜', '♝', '♞', '♟'];
  
  return (player === 'white' && whitePieces.includes(piece)) || 
         (player === 'black' && blackPieces.includes(piece));
}
  function startGame(mode) {
    welcomeScreen.style.display = "none";
    gameScreen.style.display = "flex";
    modeLabel.textContent = mode === "AI" ? "Chế độ: Máy" : "Chế độ: Người";
    playerLabel.innerHTML = `Player: ${mode === "AI" ? "🤖" : "👤"}`;
    currentPlayer = 'white';
    turnLabel.textContent = "Lượt: Trắng";
    selected = null;
    possibleMoves = [];
    renderBoard();
  }

  playAIBtn.addEventListener("click", () => startGame("AI"));
  playHumanBtn.addEventListener("click", () => startGame("Human"));
  restartBtn.addEventListener("click", () => {
    if (confirm("Bạn có chắc muốn chơi lại từ đầu không?")) {
      initialBoard.splice(
        0,
        initialBoard.length,
        ...[
          ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
          ["♟", "♟", "♟", "♟", "♟", "♟", "♟", "♟"],
          Array(8).fill(""),
          Array(8).fill(""),
          Array(8).fill(""),
          Array(8).fill(""),
          ["♙", "♙", "♙", "♙", "♙", "♙", "♙", "♙"],
          ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"],
        ]
      );
      currentPlayer = 'white';
      turnLabel.textContent = "Lượt: Trắng";
      selected = null;
      possibleMoves = [];
      renderBoard();
    }
  });

  settingsBtn.addEventListener("click", () => modal.classList.remove("hidden"));
  btnContinue.addEventListener("click", () => modal.classList.add("hidden"));
  btnHome.addEventListener("click", () => window.location.reload());
});
