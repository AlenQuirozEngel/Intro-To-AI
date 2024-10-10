#!/usr/bin/env python

"""
Runnable script. By default, plays a game where moves are selected uniformly
at random. Modified to implement AI-driven decision-making with Minimax.
"""


# import random
# from tic_tac_toe import GameState, PLAYER_1, PLAYER_2

# # Function to print the game board in a readable format
# def print_board(game_state):
#     board_symbols = {0: ".", PLAYER_1: "X", PLAYER_2: "O"}
#     board = game_state.board.reshape((3, 3))
#     print("\nCurrent board:")
#     for row in board:
#         print(" ".join(board_symbols[cell] for cell in row))
#     print("\n")

# # Minimax function to ensure optimal play
# def minimax(game_state, depth, is_maximizing):
#     if game_state.is_game_over():
#         if game_state.winner == PLAYER_1:
#             return 10 - depth  # AI wins
#         elif game_state.winner == PLAYER_2:
#             return depth - 10  # Opponent wins
#         else:
#             return 0  # Draw

#     if is_maximizing:
#         best_score = float('-inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, False)
#             game_state.undo_move(move)
#             best_score = max(score, best_score)
#         return best_score
#     else:
#         best_score = float('inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, True)
#             game_state.undo_move(move)
#             best_score = min(score, best_score)
#         return best_score

# # Function to find immediate win or block moves
# def find_immediate_move(game_state):
#     # Check for an immediate winning move
#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         if game_state.winner == PLAYER_1:
#             game_state.undo_move(move)
#             return move
#         game_state.undo_move(move)

#     # Check if we can block the opponent from winning
#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         if game_state.winner == PLAYER_2:
#             game_state.undo_move(move)
#             return move
#         game_state.undo_move(move)

#     return None

# # Function to find the best move for the AI (Player 1), prioritizing blocks and preventing loss
# def find_best_move(game_state):
#     # First, check for immediate wins or blocks
#     immediate_move = find_immediate_move(game_state)
#     if immediate_move is not None:
#         return immediate_move

#     # Enhanced blocking behavior: prioritize blocking potential threats
#     best_move = None
#     best_score = float('-inf')

#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
        
#         # After the move, check if the opponent can win in the next move
#         for opponent_move in game_state.legal_moves:
#             game_state.apply_move(opponent_move)
#             if game_state.winner == PLAYER_2:
#                 game_state.undo_move(opponent_move)
#                 game_state.undo_move(move)
#                 # Block the move that would allow the opponent to win
#                 return opponent_move
#             game_state.undo_move(opponent_move)
        
#         # Evaluate the move using Minimax if no immediate threat
#         move_score = minimax(game_state, 0, False)
#         game_state.undo_move(move)

#         if move_score > best_score:
#             best_score = move_score
#             best_move = move

#     return best_move

# # Function to calculate percentages
# def calculate_percentage(count, total):
#     return round((count / total) * 100, 2)

# # Main function to play 100 games and calculate statistics, printing loss cases
# def main():
#     player1_wins = 0
#     player2_wins = 0
#     draws = 0
#     total_score = 0

#     total_games = 100

#     for game_num in range(total_games):
#         game_state = GameState()
#         num_moves = 0
#         move_history = []

#         # Play the game
#         while not game_state.is_game_over():
#             if game_state.current_player == PLAYER_1:
#                 # AI's move using Minimax
#                 move_to_play = find_best_move(game_state)
#                 move_history.append(("AI", move_to_play))
#             else:
#                 # Random opponent move
#                 move_to_play = random.choice(game_state.legal_moves)
#                 move_history.append(("Random", move_to_play))

#             game_state.apply_move(move_to_play)
#             num_moves += 1

#         # Record results and print the game if Player 2 (Random) wins
#         if game_state.winner == PLAYER_1:
#             player1_wins += 1
#             total_score += 1
#         elif game_state.winner == PLAYER_2:
#             player2_wins += 1
#             total_score -= 3
#             print(f"Game {game_num + 1}: AI lost")
#             print_board(game_state)
#             print("Move history (AI vs Random):")
#             for move in move_history:
#                 print(move)
#         else:
#             draws += 1
#             total_score -= 1

#     # Output statistics
#     print(f"\nStatistics after {total_games} games:")
#     print(f"Player 1 (AI) Wins: {player1_wins} ({calculate_percentage(player1_wins, total_games)}%)")
#     print(f"Player 2 (Random) Wins: {player2_wins} ({calculate_percentage(player2_wins, total_games)}%)")
#     print(f"Draws: {draws} ({calculate_percentage(draws, total_games)}%)")
#     print(f"Total Score: {total_score}")

# if __name__ == "__main__":
#     main()

#----------------------------------------------------------------------------------------------------------------------------------------------

# import random
# from tic_tac_toe import GameState, PLAYER_1, PLAYER_2

# # Minimax function with corrected point assignment and proper depth handling
# def minimax(game_state, depth, is_maximizing):
#     if game_state.is_game_over():
#         if game_state.winner == PLAYER_1:
#             return 10 - depth  # AI wins, prioritize winning faster
#         elif game_state.winner == PLAYER_2:
#             return depth - 10  # Opponent wins, prioritize delaying loss
#         else:
#             return 0  # Draw, neutral score

#     if is_maximizing:
#         best_score = float('-inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, False)
#             game_state.undo_move(move)
#             best_score = max(score, best_score)
#         return best_score
#     else:
#         best_score = float('inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, True)
#             game_state.undo_move(move)
#             best_score = min(score, best_score)
#         return best_score

# # Function to find the best move for the AI (Player 1)
# def find_best_move(game_state):
#     best_move = None
#     best_score = float('-inf')

#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         move_score = minimax(game_state, 0, False)  # Start depth at 0
#         game_state.undo_move(move)

#         if move_score > best_score:
#             best_score = move_score
#             best_move = move

#     return best_move

# # Function to calculate percentages
# def calculate_percentage(count, total):
#     return round((count / total) * 100, 2)

# # Function to make both players use Minimax (for the second loop)
# def find_best_move_for_opponent(game_state):
#     best_move = None
#     best_score = float('inf')  # Minimize the AI's score (opponent is minimizing)

#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         move_score = minimax(game_state, 0, True)  # Opponent is minimizing
#         game_state.undo_move(move)

#         if move_score < best_score:
#             best_score = move_score
#             best_move = move

#     return best_move

# # Main function to play games and calculate statistics
# def main():
#     total_games = 100

#     ### 1. Testing AI (Minimax) vs Random Opponent ###
#     player1_wins = 0
#     player2_wins = 0
#     draws = 0
#     total_score = 0

#     print("\n### Testing AI (Minimax) vs Random Opponent ###\n")

#     for _ in range(total_games):
#         game_state = GameState()
#         num_moves = 0

#         # Play the game
#         while not game_state.is_game_over():
#             if game_state.current_player == PLAYER_1:
#                 # AI's move using Minimax
#                 move_to_play = find_best_move(game_state)
#             else:
#                 # Random opponent move
#                 move_to_play = random.choice(game_state.legal_moves)

#             game_state.apply_move(move_to_play)
#             num_moves += 1

#         # Record results
#         if game_state.winner == PLAYER_1:
#             player1_wins += 1
#             total_score += 1
#         elif game_state.winner == PLAYER_2:
#             player2_wins += 1
#             total_score -= 3
#         else:
#             draws += 1
#             total_score -= 1

#     # Output statistics for AI vs Random
#     print(f"\nStatistics for AI vs Random after {total_games} games:")
#     print(f"Player 1 (AI) Wins: {player1_wins} ({calculate_percentage(player1_wins, total_games)}%)")
#     print(f"Player 2 (Random) Wins: {player2_wins} ({calculate_percentage(player2_wins, total_games)}%)")
#     print(f"Draws: {draws} ({calculate_percentage(draws, total_games)}%)")
#     print(f"Total Score: {total_score}")

#     ### 2. Testing AI (Minimax) vs Minimax Opponent ###
#     player1_wins_minimax = 0
#     player2_wins_minimax = 0
#     draws_minimax = 0
#     total_score_minimax = 0

#     print("\n### Testing AI (Minimax) vs Minimax Opponent ###\n")

#     for _ in range(total_games):
#         game_state = GameState()
#         num_moves = 0

#         # Play the game (both players using Minimax)
#         while not game_state.is_game_over():
#             if game_state.current_player == PLAYER_1:
#                 # AI's move using Minimax
#                 move_to_play = find_best_move(game_state)
#             else:
#                 # Opponent also using Minimax
#                 move_to_play = find_best_move_for_opponent(game_state)

#             game_state.apply_move(move_to_play)
#             num_moves += 1

#         # Record results
#         if game_state.winner == PLAYER_1:
#             player1_wins_minimax += 1
#             total_score_minimax += 1
#         elif game_state.winner == PLAYER_2:
#             player2_wins_minimax += 1
#             total_score_minimax -= 3
#         else:
#             draws_minimax += 1
#             total_score_minimax -= 1

#     # Output statistics for AI vs Minimax Opponent
#     print(f"\nStatistics for AI vs Minimax Opponent after {total_games} games:")
#     print(f"Player 1 (AI) Wins: {player1_wins_minimax} ({calculate_percentage(player1_wins_minimax, total_games)}%)")
#     print(f"Player 2 (Minimax) Wins: {player2_wins_minimax} ({calculate_percentage(player2_wins_minimax, total_games)}%)")
#     print(f"Draws: {draws_minimax} ({calculate_percentage(draws_minimax, total_games)}%)")
#     print(f"Total Score: {total_score_minimax}")

# if __name__ == "__main__":
#     main()































# import random
# import time
# from tic_tac_toe import GameState, PLAYER_1, PLAYER_2

# # Standard Minimax function
# def minimax(game_state, depth, is_maximizing):
#     # Check if the game has ended
#     if game_state.is_game_over():
#         if game_state.winner == PLAYER_1:
#             return 10 - depth  # AI wins (favor quicker wins)
#         elif game_state.winner == PLAYER_2:
#             return depth - 10  # Opponent wins (penalize more for faster losses)
#         else:
#             return 0  # Draw

#     # Maximizing player's turn (AI - Player 1)
#     if is_maximizing:
#         best_score = float('-inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, False)
#             game_state.undo_move(move)
#             best_score = max(score, best_score)
#         return best_score

#     # Minimizing player's turn (Opponent - Player 2)
#     else:
#         best_score = float('inf')
#         for move in game_state.legal_moves:
#             game_state.apply_move(move)
#             score = minimax(game_state, depth + 1, True)
#             game_state.undo_move(move)
#             best_score = min(score, best_score)
#         return best_score

# # Function to find the best move for the AI (Player 1)
# def find_best_move(game_state):
#     best_move = None
#     best_score = float('-inf')

#     # Loop over all legal moves and use minimax to evaluate
#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         move_score = minimax(game_state, 0, False)  # Opponent minimizes next
#         game_state.undo_move(move)

#         # Choose the move with the highest score
#         if move_score > best_score:
#             best_score = move_score
#             best_move = move

#     return best_move

# # Function for Player 2 (Minimax opponent)
# def find_best_move_for_opponent(game_state):
#     best_move = None
#     best_score = float('inf')

#     # Loop over all legal moves and use minimax to evaluate
#     for move in game_state.legal_moves:
#         game_state.apply_move(move)
#         move_score = minimax(game_state, 0, True)  # AI maximizes next
#         game_state.undo_move(move)

#         # Choose the move with the lowest score (opponent minimizes)
#         if move_score < best_score:
#             best_score = move_score
#             best_move = move

#     return best_move

# # Main function to play games, calculate statistics, and measure time
# def main():
#     total_games = 100

#     ### 1. Testing AI (Minimax) vs Random Opponent ###
#     print("\n### Testing AI (Minimax) vs Random Opponent ###\n")
#     start_time = time.time()  # Start time measurement
#     play_games(total_games, False)  # Play against random opponent
#     end_time = time.time()  # End time measurement
#     print(f"Time taken for AI vs Random Opponent: {round((end_time - start_time) * 1000)} ms")  # Convert to milliseconds

#     ### 2. Testing AI (Minimax) vs Minimax Opponent ###
#     print("\n### Testing AI (Minimax) vs Minimax Opponent ###\n")
#     start_time = time.time()  # Start time measurement
#     play_games(total_games, True)  # Play against Minimax opponent
#     end_time = time.time()  # End time measurement
#     print(f"Time taken for AI vs Minimax Opponent: {round((end_time - start_time) * 1000)} ms")  # Convert to milliseconds

# def play_games(total_games, use_minimax_opponent):
#     # Variables to keep track of wins, draws, and score
#     player1_wins = 0
#     player2_wins = 0
#     draws = 0
#     total_score = 0

#     for game_num in range(total_games):
#         game_state = GameState()

#         while not game_state.is_game_over():
#             if game_state.current_player == PLAYER_1:
#                 # AI's move (using Minimax)
#                 move_to_play = find_best_move(game_state)
#             else:
#                 if use_minimax_opponent:
#                     # Opponent's move using Minimax
#                     move_to_play = find_best_move_for_opponent(game_state)
#                 else:
#                     # Random opponent's move
#                     move_to_play = random.choice(game_state.legal_moves)

#             game_state.apply_move(move_to_play)

#         # Track game results
#         if game_state.winner == PLAYER_1:
#             player1_wins += 1
#             total_score += 1  # AI wins
#         elif game_state.winner == PLAYER_2:
#             player2_wins += 1
#             total_score -= 3  # AI loses
#         else:
#             draws += 1
#             total_score -= 1  # Draw

#     # Output statistics after the games
#     print(f"Statistics after {total_games} games:")
#     print(f"Player 1 (AI) Wins: {player1_wins} ({round(player1_wins / total_games * 100, 2)}%)")
#     print(f"Player 2 {('(Minimax)' if use_minimax_opponent else '(Random)')} Wins: {player2_wins} ({round(player2_wins / total_games * 100, 2)}%)")
#     print(f"Draws: {draws} ({round(draws / total_games * 100, 2)}%)")
#     print(f"Total Score: {total_score}")

# if __name__ == "__main__":
#     main()

import random

class GameState:
    
    PLAYER_1 = 1  # AI
    PLAYER_2 = -1  # Opponent
    NUM_ROWS = 3
    NUM_COLS = 3
    
    # Predefined lines to check for win conditions
    LINES_TO_CHECK = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical
        [0, 4, 8], [2, 4, 6]              # Diagonal
    ]
    
    def __init__(self):
        # Initialize the game board (empty)
        self.board = [0] * (self.NUM_ROWS * self.NUM_COLS)
        self.legal_moves = list(range(9))  # All moves are legal at the start
        self.current_player = self.PLAYER_1  # AI starts
        self.winner = 0  # No winner initially

    def get_legal_moves(self):
        return self.legal_moves[:]

    def apply_move(self, move):
        # Apply the move for the current player
        self.board[move] = self.current_player
        self.legal_moves.remove(move)

        # Check for a win
        for line in self.LINES_TO_CHECK:
            if all(self.board[cell] == self.current_player for cell in line):
                self.winner = self.current_player
                break

        # Swap players
        self.current_player *= -1

    def undo_move(self, move):
        # Undo the move
        self.board[move] = 0
        self.legal_moves.append(move)
        self.winner = 0  # Reset the winner after undoing
        self.current_player *= -1

    def is_game_over(self):
        return self.winner != 0 or len(self.legal_moves) == 0

    def get_winner(self):
        return self.winner

    def get_current_player(self):
        return self.current_player


# Minimax Algorithm with Alpha-Beta Pruning
def minimax(game_state, depth, alpha, beta, is_maximizing):
    if game_state.is_game_over():
        if game_state.get_winner() == GameState.PLAYER_1:
            return 10  # AI wins
        elif game_state.get_winner() == GameState.PLAYER_2:
            return -10  # Opponent wins
        else:
            return 0  # Draw

    legal_moves = game_state.get_legal_moves()

    if is_maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            game_state.apply_move(move)
            eval = minimax(game_state, depth + 1, alpha, beta, False)
            game_state.undo_move(move)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            game_state.apply_move(move)
            eval = minimax(game_state, depth + 1, alpha, beta, True)
            game_state.undo_move(move)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


# Find the best move using Minimax with Alpha-Beta Pruning
def find_best_move(game_state, is_maximizing):
    best_move = -1
    best_score = float('-inf') if is_maximizing else float('inf')

    legal_moves = game_state.get_legal_moves()

    for move in legal_moves:
        game_state.apply_move(move)
        move_score = minimax(game_state, 0, float('-inf'), float('inf'), not is_maximizing)
        game_state.undo_move(move)

        if is_maximizing:
            if move_score > best_score:
                best_score = move_score
                best_move = move
        else:
            if move_score < best_score:
                best_score = move_score
                best_move = move

    return best_move


# Play multiple games between AI and either Random or Minimax opponent
def play_games(total_games, use_minimax_opponent):
    player1_wins = 0
    player2_wins = 0
    draws = 0
    total_score = 0

    for _ in range(total_games):
        game_state = GameState()

        while not game_state.is_game_over():
            if game_state.get_current_player() == GameState.PLAYER_1:
                # AI's move using Minimax
                move_to_play = find_best_move(game_state, True)  # AI maximizes
            else:
                if use_minimax_opponent:
                    # Opponent's move using Minimax
                    move_to_play = find_best_move(game_state, False)  # Opponent minimizes
                else:
                    # Random opponent's move
                    legal_moves = game_state.get_legal_moves()
                    move_to_play = random.choice(legal_moves)

            game_state.apply_move(move_to_play)

        # Track results
        if game_state.get_winner() == GameState.PLAYER_1:
            player1_wins += 1
            total_score += 1
        elif game_state.get_winner() == GameState.PLAYER_2:
            player2_wins += 1
            total_score -= 3
        else:
            draws += 1
            total_score -= 1

    print(f"Statistics after {total_games} games:")
    print(f"Player 1 (AI) Wins: {player1_wins} ({player1_wins / total_games * 100:.2f}%)")
    print(f"Player 2 {'(Minimax)' if use_minimax_opponent else '(Random)'} Wins: {player2_wins} ({player2_wins / total_games * 100:.2f}%)")
    print(f"Draws: {draws} ({draws / total_games * 100:.2f}%)")
    print(f"Total Score: {total_score}")


# Main Function
if __name__ == "__main__":
    total_games = 100

    print("### Testing AI (Minimax) vs Random Opponent ###")
    play_games(total_games, False)  # AI vs Random opponent

    print("\n### Testing AI (Minimax) vs Minimax Opponent ###")
    play_games(total_games, True)  # AI vs Minimax opponent
