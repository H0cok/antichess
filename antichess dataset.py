import numpy
import chess
import chess.engine
import chess.variant
import chess.pgn as chp
import numpy as np
import pickle




# example: h3 -> 17
def square_to_index(square, squares_index):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board, squares_index):
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
        i, j = square_to_index(move.to_square, squares_index)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square, squares_index)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d


def stockfish(board, depth, sf):
    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white()
    if score.is_mate():
        return int(f"{score.__str__()[1]}1000")
    else:
        return int(score.__str__())


def dataset(file_name, depth):
    with chess.engine.SimpleEngine.popen_uci('antichess_engine') as sf:
        pgn = open(file_name)
        dataset = [[], []]
        pickl = open("file.pickle", "rb")
        pos = pickle.load(pickl)
        print(type(pos))
        pickl.close()
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

        for i in range(1000):
            if i <= 100:
                continue
            game = chp.read_game(pgn)

            if not game:
                return dataset
            board = game.board()
            for move in game.mainline_moves():
                if len(board.move_stack) < 22:
                    if board.fen().__str__() in pos:
                        pass
                    else:
                        dataset[0].append(split_dims(board, squares_index))
                        a = stockfish(board, depth=depth, sf=sf)
                        pos[board.fen().__str__()] = a
                        dataset[1].append(a)
                else:
                    dataset[0].append(split_dims(board, squares_index))
                    dataset[1].append(stockfish(board, depth=depth, sf=sf))
                board.push(move)
            print(i)
        return dataset, pos


a, pos = dataset("lichess.pgn", 16)
print(pos)
print(len(a[0]), len(a[1]))

np.savez("go100_1000.npz", b=a[0], v=a[1])

f = open("file.pickle", "wb")

# write the python object (dict) to pickle file
pickle.dump(pos, f, protocol=pickle.HIGHEST_PROTOCOL)

# close file
f.close()
