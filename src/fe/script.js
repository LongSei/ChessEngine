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

// Utility: generate possible moves (basic logic)
function getPossibleMoves(r, c) {
  const piece = initialBoard[r][c];
  const moves = [];
  switch (piece) {
    case "♙": // white pawn
      if (r > 0 && !initialBoard[r - 1][c]) moves.push([r - 1, c]);
      break;
    case "♟": // black pawn
      if (r < 7 && !initialBoard[r + 1][c]) moves.push([r + 1, c]);
      break;
    case "♘":
    case "♞": // knight
      [
        [2, 1],
        [2, -1],
        [-2, 1],
        [-2, -1],
        [1, 2],
        [1, -2],
        [-1, 2],
        [-1, -2],
      ].forEach((d) => {
        const nr = r + d[0],
          nc = c + d[1];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) moves.push([nr, nc]);
      });
      break;
    // TODO: add other pieces (rook, bishop, queen, king)
    default:
      break;
  }
  return moves;
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
          if (p) cell.textContent = p;
          cell.addEventListener("click", () => handleCellClick(r, c - 1));
          if (selected && selected.row === r && selected.col === c - 1)
            cell.classList.add("selected");
          possibleMoves.forEach((m) => {
            if (m[0] === r && m[1] === c - 1) cell.classList.add("possible");
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
    if (selected) {
      // move to clicked if in possibleMoves
      if (possibleMoves.some((m) => m[0] === row && m[1] === col)) {
        const { row: sr, col: sc } = selected;
        initialBoard[row][col] = initialBoard[sr][sc];
        initialBoard[sr][sc] = "";
      }
      selected = null;
      possibleMoves = [];
      renderBoard();
    } else if (piece) {
      selected = { row, col };
      possibleMoves = getPossibleMoves(row, col);
      renderBoard();
    }
  }

  function startGame(mode) {
    welcomeScreen.style.display = "none";
    gameScreen.style.display = "flex";
    modeLabel.textContent = mode === "AI" ? "Chế độ: Máy" : "Chế độ: Người";
    playerLabel.innerHTML = `Player: ${mode === "AI" ? "🤖" : "👤"}`;
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
