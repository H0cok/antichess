import chess
import chess.engine
import numpy
import tensorflow.keras.models as models
import chess.svg
import chess.variant


def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci('antichess_engine') as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white()
        if score.is_mate():
            return int(f"{score.__str__()[1]}1000")
        else:
            return int(score.__str__())


squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}


def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d matrix
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    # here we add the pieces's view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d


model = models.load_model("model0_10000_32_4.h5")
model.summary()


# used for the minimax algorithm
def minimax_eval(board):
    board3d = split_dims(board)
    board3d = numpy.expand_dims(board3d, 0)
    return model(board3d)[0][0]


def minimax(board, depth, alpha, beta, maximizing_player, real_depth):


    if depth == 0 or board.is_game_over() or real_depth>=15:
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


# this is the actual function that gets the move from the neural network
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
