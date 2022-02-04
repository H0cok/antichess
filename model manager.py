import chess
import chess.engine
import numpy
import tensorflow.keras.models as models
import chess.svg
import chess.variant
from creating_model import squares_index, square_to_index, split_dims

#loading model
model = models.load_model("model0_10000_32_4.h5")
model.summary()


# returns integer value (eval of the position)
def minimax_eval(board):
    board3d = split_dims(board)
    board3d = numpy.expand_dims(board3d, 0)
    return model(board3d)[0][0]

# improved minmax algorithm (if the amount of new brunches is <=4 it goes deeper)
def minimax(board, depth, alpha, beta, maximizing_player, real_depth):
    if depth == 0 or board.is_game_over() or real_depth >= 10:
        return minimax_eval(board)
    if maximizing_player:
        max_eval = -numpy.inf
        for move in board.legal_moves:
            board.push(move)
            if len(list(board.legal_moves)) <= 4:
                depth_new = depth
                real_depth = real_depth +1
            else:
                depth_new = depth - 1
            eval = minimax(board, depth_new, alpha, beta, False, real_depth)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = numpy.inf
        for move in board.legal_moves:
            board.push(move)
            if len(list(board.legal_moves)) <= 4:
                depth_new = depth
                real_depth = real_depth + 1
            else:
                depth_new = depth - 1
            eval = minimax(board, depth_new, alpha, beta, True, real_depth)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


# gets the move from the neural network
def get_ai_move(board, depth):
    max_move = None
    max_eval = -numpy.inf
    for move in board.legal_moves:
        board.push(move)
        moves = []
        if len(list(board.legal_moves)) <= 3:
            depth_new = depth
        else:
            depth_new = depth - 1
        eval = minimax(board, depth_new, -numpy.inf, numpy.inf, False, 1)
        print(move, eval)
        board.pop()
    return max_move

# setting board with possible custom starting possition
board = chess.variant.AntichessBoard("rn1qkbnr/pp1bpp1p/8/6B1/8/2P3P1/P1P1PP1P/R3KBNR b - - 0 6")
while 1:
    a = len(list(board.legal_moves))
    if a == 1:
        move = get_ai_move(board, 1)
    else:
        move = get_ai_move(board, 4)
    board.push(move)
    print(board)
    move = chess.Move.from_uci(input())
    board.push(move)
    print(board)
