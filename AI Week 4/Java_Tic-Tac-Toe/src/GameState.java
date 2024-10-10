import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GameState {
    
    public static final int PLAYER_1 = 1;  // AI
    public static final int PLAYER_2 = -1; // Opponent
    
    private static final int NUM_ROWS = 3;
    private static final int NUM_COLS = 3;
    
    private static final int[][][] LINES_TO_CHECK;
    
    // Game state variables
    private final int[] board;
    private final List<Integer> legalMoves;
    private int currentPlayer;
    private int winner;

    /**
     * Creates a new initial game state (with an empty board).
     */
    public GameState() {
        board = new int[NUM_ROWS * NUM_COLS];

        legalMoves = new ArrayList<>(9);
        for (int i = 0; i < 9; ++i) {
            legalMoves.add(i);
        }

        currentPlayer = PLAYER_1;  // AI starts
        winner = 0;
    }

    /**
     * @return An unmodifiable view of the list of legal moves in the current game state.
     */
    public List<Integer> getLegalMoves() {
        return Collections.unmodifiableList(legalMoves);
    }

    /**
     * Modifies the game state by applying the given move.
     * 
     * @param move A 0-based index representing where we want to place our piece.
     */
    public void applyMove(final int move) {
        board[move] = currentPlayer;  // Set the move
        Utils.removeSwap(legalMoves, legalMoves.indexOf(move));  // Remove the move from the list of legal moves

        // Check if the current player won
        final int[][] linesToCheck = LINES_TO_CHECK[move];
        for (final int[] line : linesToCheck) {
            boolean win = true;
            for (final int cell : line) {
                if (board[cell] != currentPlayer) {
                    win = false;
                    break;
                }
            }
            if (win) {
                winner = currentPlayer;
                break;
            }
        }

        // Swap players after the move
        currentPlayer *= -1;
    }

    /**
     * Reverts the game state back to the state before the given move was played.
     * 
     * @param move The move to be undone.
     */
    public void undoMove(final int move) {
        board[move] = 0;  // Reset the move
        legalMoves.add(move);  // Add the move back to legal moves
        winner = 0;  // Reset winner
        currentPlayer *= -1;  // Swap the player back
    }

    /**
     * @return True if the game is over, false otherwise.
     */
    public boolean isGameOver() {
        return winner != 0 || legalMoves.isEmpty();
    }

    /**
     * @return The winner (1 for PLAYER_1, -1 for PLAYER_2, 0 for no winner).
     */
    public int getWinner() {
        return winner;
    }

    /**
     * Getter for the current player.
     * 
     * @return The current player (PLAYER_1 or PLAYER_2).
     */
    public int getCurrentPlayer() {
        return currentPlayer;
    }

    /**
     * Converts a board index into a row and column.
     * 
     * @param index A 0-based index.
     * @return The [row, column] corresponding to the given index.
     */
    public static int[] indexToRowCol(final int index) {
        return new int[] {index / NUM_COLS, index % NUM_COLS};
    }

    /**
     * Converts a row and column into a board index.
     * 
     * @param rowCol The [row, column].
     * @return A 0-based index corresponding to the given row and column.
     */
    public static int rowColToIndex(final int[] rowCol) {
        return (rowCol[0] * NUM_COLS) + rowCol[1];
    }

    // Pre-compute the lines to check for potential wins for each move
    static {
        LINES_TO_CHECK = new int[NUM_COLS * NUM_ROWS][][];

        for (int i = 0; i < LINES_TO_CHECK.length; ++i) {
            final int[] rowCol = indexToRowCol(i);
            final int row = rowCol[0];
            final int col = rowCol[1];

            int numLinesToCheck = 2;

            if (row == 0 && col != 1) ++numLinesToCheck;  // row-0 corner
            else if (row == 2 && col != 1) ++numLinesToCheck;  // row-2 corner
            else if (row == 1 && col == 1) numLinesToCheck += 2;  // center square

            LINES_TO_CHECK[i] = new int[numLinesToCheck][2];

            // Horizontal line
            int nextIdx = 0;
            if (col != 0) LINES_TO_CHECK[i][0][nextIdx++] = rowColToIndex(new int[] {row, 0});
            if (col != 1) LINES_TO_CHECK[i][0][nextIdx++] = rowColToIndex(new int[] {row, 1});
            if (col != 2) LINES_TO_CHECK[i][0][nextIdx++] = rowColToIndex(new int[] {row, 2});

            // Vertical line
            nextIdx = 0;
            if (row != 0) LINES_TO_CHECK[i][1][nextIdx++] = rowColToIndex(new int[] {0, col});
            if (row != 1) LINES_TO_CHECK[i][1][nextIdx++] = rowColToIndex(new int[] {1, col});
            if (row != 2) LINES_TO_CHECK[i][1][nextIdx++] = rowColToIndex(new int[] {2, col});

            // Diagonals (only for corners and center)
            if (col != 1 && (row == 0 || row == 2)) {
                if (col == 0) {
                    LINES_TO_CHECK[i][2][0] = rowColToIndex(new int[] {1, 1});
                    LINES_TO_CHECK[i][2][1] = rowColToIndex(new int[] {2, 2});
                } else if (col == 2) {
                    LINES_TO_CHECK[i][2][0] = rowColToIndex(new int[] {1, 1});
                    LINES_TO_CHECK[i][2][1] = rowColToIndex(new int[] {2, 0});
                }
            } else if (row == 1 && col == 1) {
                LINES_TO_CHECK[i][2][0] = rowColToIndex(new int[] {0, 0});
                LINES_TO_CHECK[i][2][1] = rowColToIndex(new int[] {2, 2});

                LINES_TO_CHECK[i][3][0] = rowColToIndex(new int[] {0, 2});
                LINES_TO_CHECK[i][3][1] = rowColToIndex(new int[] {2, 0});
            }
        }
    }
}
