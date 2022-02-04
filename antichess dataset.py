import numpy
import chess
import chess.engine
import chess.variant
import chess.pgn as chp
import numpy as np
import pickle
from creating_model import squares_index, square_to_index, split_dims

#using stockfish to eval the position
def stockfish(board, depth, sf):
    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white()
    if score.is_mate():
        return int(f"{score.__str__()[1]}1000")
    else:
        return int(score.__str__())

# function adds new values to dataset
def dataset(file_name, depth):
    with chess.engine.SimpleEngine.popen_uci('antichess_engine') as sf:
        pgn = open(file_name)
        dataset = [[], []]
        # opening file with already evaluated positions
        pickl = open("file.pickle", "rb")
        pos = pickle.load(pickl)
        pickl.close()
        for i in range(10000):
            if i <= 100:
                continue
            # getting new game from dataset
            game = chp.read_game(pgn)
            # in case there are no more games in png
            if not game:
                return dataset
            board = game.board()
            for move in game.mainline_moves():
                if len(board.move_stack) < 22:
                    # remembering already evaluated possitions
                    if board.fen().__str__() in pos:
                        pass
                    else:
                        dataset[0].append(split_dims(board))
                        a = stockfish(board, depth=depth, sf=sf)
                        pos[board.fen().__str__()] = a
                        dataset[1].append(a)
                else:
                    dataset[0].append(split_dims(board))
                    dataset[1].append(stockfish(board, depth=depth, sf=sf))
                board.push(move)
            # just to ensure that everything is working
            print(i)
        return dataset, pos


a, pos = dataset("lichess.pgn", 16)
print(pos)
print(len(a[0]), len(a[1]))
np.savez("go100_1000.npz", b=a[0], v=a[1])
f = open("file.pickle", "wb")
# write the python object (dict) to pickle file to save already evaluated positions
pickle.dump(pos, f, protocol=pickle.HIGHEST_PROTOCOL)
# close file
f.close()
