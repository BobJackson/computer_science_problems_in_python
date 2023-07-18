from chapter08.board import Board, Move
from chapter08.connectfour import C4Board
from chapter08.minimax import find_best_move

board: Board = C4Board()


def get_player_move() -> Move:
    player_move: Move = Move(-1)
    while player_move not in board.legal_moves:
        play: int = int(input("Enter a legal column (0-6):"))
        player_move = Move(play)
    return player_move


if __name__ == '__main__':
    while True:
        human_move: Move = get_player_move()
        board = board.move(human_move)
        if board.is_win:
            print("Human wins!")
            break
        elif board.is_draw:
            print("Draw!")
            break
        compute_move: Move = find_best_move(board, 3)
        print(f"Computer move is {compute_move}")
        board = board.move(compute_move)
        print(board)
        if board.is_win:
            print("Computer wins!")
            break
        elif board.is_draw:
            print("Draw!")
            break
