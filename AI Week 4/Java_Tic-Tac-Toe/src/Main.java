// import java.util.List;
// import java.util.concurrent.ThreadLocalRandom;

// public class Main {

//     public static void main(final String[] args) {
//         // Variables to keep track of wins, draws, and score
//         int player1Wins = 0;
//         int player2Wins = 0;
//         int draws = 0;
//         int totalScore = 0;  // Total score to keep track of the AI's performance

//         // Play 1000 games
//         int totalGames = 1000;
//         for (int i = 0; i < totalGames; i++) {
//             GameState gameState = new GameState();
//             int numMoves = 0;

//             // Play a single game
//             while (!gameState.isGameOver()) {
//                 int moveToPlay;

//                 if (gameState.getCurrentPlayer() == GameState.PLAYER_1) {
//                     // AI's move (using Minimax)
//                     moveToPlay = findBestMove(gameState);
//                 } else {
//                     // Opponent's move (random)
//                     final List<Integer> legalMoves = gameState.getLegalMoves();
//                     moveToPlay = legalMoves.get(ThreadLocalRandom.current().nextInt(legalMoves.size()));
//                 }

//                 gameState.applyMove(moveToPlay);
//                 ++numMoves;
//             }

//             // Track results of the game
//             if (gameState.getWinner() == GameState.PLAYER_1) {
//                 player1Wins++;
//                 totalScore += 1;  // AI wins, +1 score
//             } else if (gameState.getWinner() == GameState.PLAYER_2) {
//                 player2Wins++;
//                 totalScore -= 3;  // AI loses, -3 score
//             } else {
//                 draws++;
//                 totalScore -= 1;  // Draw, -1 score
//             }
//         }

//         // Output statistics after 1000 games
//         System.out.println("Statistics after " + totalGames + " games:");
//         System.out.println("Player 1 (AI) Wins: " + player1Wins + " (" + getPercentage(player1Wins, totalGames) + "%)");
//         System.out.println("Player 2 (Random) Wins: " + player2Wins + " (" + getPercentage(player2Wins, totalGames) + "%)");
//         System.out.println("Draws: " + draws + " (" + getPercentage(draws, totalGames) + "%)");
//         System.out.println("Total Score: " + totalScore);
//     }

//     // Minimax algorithm implementation
//     public static int minimax(GameState gameState, boolean isMaximizing) {
//         // Base case: If the game is over, return the score
//         if (gameState.isGameOver()) {
//             if (gameState.getWinner() == GameState.PLAYER_1) {
//                 return 1;  // AI wins
//             } else if (gameState.getWinner() == GameState.PLAYER_2) {
//                 return -3; // Opponent wins
//             } else {
//                 return -1; // Draw
//             }
//         }

//         // Copy of legal moves to avoid ConcurrentModificationException
//         List<Integer> legalMovesCopy = List.copyOf(gameState.getLegalMoves());

//         if (isMaximizing) {
//             int bestScore = Integer.MIN_VALUE;
//             for (int move : legalMovesCopy) {
//                 gameState.applyMove(move);
//                 int score = minimax(gameState, false); // Opponent's turn next
//                 gameState.undoMove(move); // Undo move to backtrack
//                 bestScore = Math.max(score, bestScore); // Get the highest score
//             }
//             return bestScore;
//         } else {
//             int bestScore = Integer.MAX_VALUE;
//             for (int move : legalMovesCopy) {
//                 gameState.applyMove(move);
//                 int score = minimax(gameState, true); // AI's turn next
//                 gameState.undoMove(move); // Undo move to backtrack
//                 bestScore = Math.min(score, bestScore); // Get the lowest score
//             }
//             return bestScore;
//         }
//     }

//     // Method to find the best move for AI (Player 1)
//     public static int findBestMove(GameState gameState) {
//         int bestMove = -1;
//         int bestScore = Integer.MIN_VALUE;

//         // Copy of legal moves to avoid ConcurrentModificationException
//         List<Integer> legalMovesCopy = List.copyOf(gameState.getLegalMoves());

//         // Iterate over all legal moves
//         for (int move : legalMovesCopy) {
//             gameState.applyMove(move); // Try the move
//             int moveScore = minimax(gameState, false); // Evaluate it using minimax
//             gameState.undoMove(move); // Undo the move after evaluating

//             if (moveScore > bestScore) {
//                 bestScore = moveScore;
//                 bestMove = move; // Select the move with the highest score
//             }
//         }

//         return bestMove;
//     }

//     // Utility function to calculate percentage
//     public static String getPercentage(int count, int total) {
//         return String.format("%.2f", (count * 100.0) / total);
//     }
// }
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class Main {

    public static void main(final String[] args) {
        int totalGames = 100;

        // First loop: AI (Minimax) vs Random Opponent
        System.out.println("### Testing AI (Minimax) vs Random Opponent ###");
        long startTime = System.currentTimeMillis();
        playGames(totalGames, false); // Play against random opponent
        long endTime = System.currentTimeMillis();
        System.out.println("Time taken for AI vs Random Opponent: " + (endTime - startTime) + " ms");

        // Second loop: AI (Minimax) vs Minimax Opponent
        System.out.println("\n### Testing AI (Minimax) vs Minimax Opponent ###");
        startTime = System.currentTimeMillis();
        playGames(totalGames, true); // Play against Minimax opponent
        endTime = System.currentTimeMillis();
        System.out.println("Time taken for AI vs Minimax Opponent: " + (endTime - startTime) + " ms");
    }

    public static void playGames(int totalGames, boolean useMinimaxOpponent) {
        // Variables to keep track of wins, draws, and score
        int player1Wins = 0;
        int player2Wins = 0;
        int draws = 0;
        int totalScore = 0;  // Total score to keep track of the AI's performance

        for (int i = 0; i < totalGames; i++) {
            GameState gameState = new GameState();
            int numMoves = 0;

            // Play a single game
            while (!gameState.isGameOver()) {
                int moveToPlay;

                if (gameState.getCurrentPlayer() == GameState.PLAYER_1) {
                    // AI's move (using Minimax)
                    moveToPlay = findBestMove(gameState, true);  // Player 1 maximizes
                } else {
                    if (useMinimaxOpponent) {
                        // Opponent's move using Minimax
                        moveToPlay = findBestMove(gameState, false); // Player 2 minimizes
                    } else {
                        // Random opponent's move
                        final List<Integer> legalMoves = gameState.getLegalMoves();
                        moveToPlay = legalMoves.get(ThreadLocalRandom.current().nextInt(legalMoves.size()));
                    }
                }

                gameState.applyMove(moveToPlay);
                ++numMoves;
            }

            // Track results of the game
            if (gameState.getWinner() == GameState.PLAYER_1) {
                player1Wins++;
                totalScore += 1;  // AI wins, +1 score
            } else if (gameState.getWinner() == GameState.PLAYER_2) {
                player2Wins++;
                totalScore -= 3;  // AI loses, -3 score
            } else {
                draws++;
                totalScore -= 1;  // Draw, -1 score
            }
        }

        // Output statistics after the games
        System.out.println("Statistics after " + totalGames + " games:");
        System.out.println("Player 1 (AI) Wins: " + player1Wins + " (" + getPercentage(player1Wins, totalGames) + "%)");
        System.out.println("Player 2 " + (useMinimaxOpponent ? "(Minimax)" : "(Random)") + " Wins: " + player2Wins + " (" + getPercentage(player2Wins, totalGames) + "%)");
        System.out.println("Draws: " + draws + " (" + getPercentage(draws, totalGames) + "%)");
        System.out.println("Total Score: " + totalScore);
    }

    // Minimax algorithm implementation with both maximizing and minimizing
    public static int minimax(GameState gameState, boolean isMaximizing) {
        // Base case: If the game is over, return the score
        if (gameState.isGameOver()) {
            if (gameState.getWinner() == GameState.PLAYER_1) {
                return 10;  // AI wins
            } else if (gameState.getWinner() == GameState.PLAYER_2) {
                return -10; // Opponent wins
            } else {
                return 0; // Draw
            }
        }

        List<Integer> legalMovesCopy = List.copyOf(gameState.getLegalMoves());

        if (isMaximizing) {
            int bestScore = Integer.MIN_VALUE;
            for (int move : legalMovesCopy) {
                gameState.applyMove(move);
                int score = minimax(gameState, false); // Opponent's turn next
                gameState.undoMove(move); // Undo move to backtrack
                bestScore = Math.max(score, bestScore); // Maximize AI's score
            }
            return bestScore;
        } else {
            int bestScore = Integer.MAX_VALUE;
            for (int move : legalMovesCopy) {
                gameState.applyMove(move);
                int score = minimax(gameState, true); // AI's turn next
                gameState.undoMove(move); // Undo the move to backtrack
                bestScore = Math.min(score, bestScore); // Minimize opponent's score
            }
            return bestScore;
        }
    }

    // Method to find the best move for either player (Player 1 maximizes, Player 2 minimizes)
    public static int findBestMove(GameState gameState, boolean isMaximizing) {
        int bestMove = -1;
        int bestScore = isMaximizing ? Integer.MIN_VALUE : Integer.MAX_VALUE;

        List<Integer> legalMovesCopy = List.copyOf(gameState.getLegalMoves());

        // Iterate over all legal moves
        for (int move : legalMovesCopy) {
            gameState.applyMove(move); // Try the move
            int moveScore = minimax(gameState, !isMaximizing); // Minimax based on next player's turn
            gameState.undoMove(move); // Undo the move after evaluating

            if (isMaximizing) {
                if (moveScore > bestScore) {
                    bestScore = moveScore;
                    bestMove = move; // Select the move with the highest score (maximize)
                }
            } else {
                if (moveScore < bestScore) {
                    bestScore = moveScore;
                    bestMove = move; // Select the move with the lowest score (minimize)
                }
            }
        }

        return bestMove;
    }

    // Utility function to calculate percentage
    public static String getPercentage(int count, int total) {
        return String.format("%.2f", (count * 100.0) / total);
    }
}
