import numpy as np


tetraminos = {
    "Tu": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Tr": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Td": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Tl": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # J
    "Jl": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Ju": np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Jr": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Jd": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # Z
    "Zh": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Zv": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # O
    "O":  np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # S
    "Sh": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Sv": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # L
    "Lr": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Ld": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Ll": np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Lu": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    # I
    "Iv": np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),

    "Ih": np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8),
}


class Grid:

    def __init__(self, size=(20,10)):
        self._grid = np.zeros(size, dtype=np.uint8)

    def __init__(self, grid):
        self._grid = grid

    def apply_shape(self, shape, pos):
        # row, col = shape.coords()
        # w, h = row+5, col+5
        # self._grid[row:w, col:h] = shape.data()
        for row in range(5):
            for col in range(5):
                v = shape[row, col]
                if v:
                    self._grid[max(0, pos[0] + row), max(0, pos[1] + col)] = v

    def remove_shape(self, shape, pos):
        # row, col = shape.coords()
        # w, h = row+5, col+5
        # self._grid[row:w, col:h] *= np.zeros_like(shape.data())
        for row in range(5):
            for col in range(5):
                if shape[row, col]:
                    self._grid[max(0, pos[0] + row), max(0, pos[1] + col)] = 0

    def collides(self, shape, pos):
        # row, col = shape.coords()
        # w, h = row+5, col+5
        # grid = self._grid.copy()
        # print(grid[row:w, col:h])
        # grid[row:w, col:h] *= shape.data()
        # return np.any(grid[row:w, col:h])
        for row in range(5):
            for col in range(5):
                if shape[row, col]:
                    r = pos[0] + row
                    c = pos[1] + col
                    r_indx = max(0, r)
                    c_indx = max(0, c)
                    if r >= self.rows() or c >= self.cols() or c < 0 or self._grid[r_indx, c_indx] != 0:
                        return True

        return False

    def rows(self):
        return self._grid.shape[0]

    def cols(self):
        return self._grid.shape[1]

    def __repr__(self):
        return f"{self._grid}"

    def get_peaks(self):
        peaks = [self.rows()] * self.cols()
        for col in range(self.cols()):
            for row in range(self.rows()):
                if self._grid[row, col] != 0 and peaks[col] == self.rows():
                    peaks[col] = row
        return peaks

    def get_cumulative_hight(self):
        peaks = self.get_peaks()
        total_height = [self.rows() - peak for peak in peaks]
        return sum(total_height)

    def get_holes(self):
        peaks = self.get_peaks()
        holes = 0
        for col in range(self.cols()):
            for row in range(peaks[col], self.rows()):
                if self._grid[row, col] == 0:
                    holes += 1
        return holes

    def get_clear_rows(self):
        clear_rows = 0
        for row in range(self.rows()):
            holes = 0
            for col in range(self.cols()):
                if self._grid[row, col] == 0:
                    holes += 1
            if holes == 0:
                clear_rows += 1
        return clear_rows

    def get_roughness(self):
        peaks = self.get_peaks()
        roughness = 0.
        for i in range(0, len(peaks)-1):
            roughness += abs(peaks[i] - peaks[i+1])
        return roughness

    def get_relative_height(self):
        peaks = self.get_peaks()
        # max_peak = np.argmax(peaks)
        # min_peak = np.argmin(peaks)
        return max(peaks) - min(peaks)

    def get_wheight(self):
        peaks = self.get_peaks()
        height = self.rows() - min(peaks)
        return pow(height, 1.5)

    def data(self):
        return self._grid

    def get_rating(self, genome):
        cum_height = self.get_cumulative_hight()
        holes = self.get_holes()
        rows_cleared = self.get_clear_rows()
        roughness = self.get_roughness()
        rel_height = self.get_relative_height()
        w_height = self.get_wheight()

        rating = 0.
        rating += genome["rowsCleared"] * rows_cleared
        rating += genome["weightedHeight"] * w_height
        rating += genome["cumulativeHeight"] * cum_height
        rating += genome["relativeHeight"] * rel_height
        rating += genome["holes"] * holes
        rating += genome["roughness"] * roughness
        return rating

    def drop(self, piece, starting_pos):
        pos = starting_pos
        while True:
            pos[0] += 1
            if self.collides(piece, pos):
                pos[0] -= 1
                break
        self.apply_shape(piece, pos)

class Shape:
    def __init__(self, row, col, shape):
        self.row = row
        self.col = col
        self.shape = tetraminos[shape]

    def move_left(self, grid):
        grid.remove_shape(self)
        self.col -= 1
        if grid.collides(self):
            self.col += 1
        grid.apply_shape(self)

    def move_right(self, grid):
        grid.remove_shape(self)
        self.col += 1
        if grid.collides(self):
            self.col -= 1
        grid.apply_shape(self)

    def move_down(self, grid):
        grid.remove_shape(self)
        self.row += 1
        if grid.collides(self):
            self.row -= 1
        grid.apply_shape(self)

    def drop(self, grid):
        grid.remove_shape(self)
        while True:
            self.row += 1
            if grid.collides(self):
                self.row -= 1
                break
        grid.apply_shape(self)



    def rotate(self, grid):
        grid.remove_shape(self)
        self.shape = np.rot90(self.shape, 1)
        if grid.collides(self):
            self.shape = np.rot90(self.shape, -1)
        grid.apply_shape(self)

    def data(self):
        return self.shape

    # def row(self):
    #     return self.row

    # def col(self):
    #     return self.col

    def coords(self):
        return self.row, self.col

    def __repr__(self):
        return f"{self.shape}"


from copy import copy

def getAllPossibleMoves(_grid : Grid, _shape : Shape, genome):
    possible_moves = []


    for rotations in range(0, 5):
        for translations in range(-5, 6):
            grid = copy(_grid)
            shape = copy(_shape)

            for j in range(0, rotations):
                shape.rotate(grid)

            if translations < 0:
                for l in range(0, abs(translations)):
                    shape.move_left(grid)
            elif translations > 0:
                for r in range(0, translations):
                    shape.move_right(grid)

            shape.drop(grid)


            # peaks = grid.get_peaks()
            cum_height = grid.get_cumulative_hight()
            holes = grid.get_holes()
            rows_cleared = grid.get_clear_rows()
            roughness = grid.get_roughness()
            rel_height = grid.get_relative_height()
            w_height = grid.get_wheight()

            rating = 0.
            rating += genome["rowsCleared"] * rows_cleared
            rating += genome["weightedHeight"] * w_height
            rating += genome["cumulativeHeight"] * cum_height
            rating += genome["relativeHeight"] * rel_height
            rating += genome["holes"] * holes
            rating += genome["roughness"] * roughness

            grid.remove_shape(shape)

            possible_moves.append((rotations, translations, rating, genome))

    return possible_moves

def piece_rotation(name):
    if name[0] in ("T", "J", "L"):
        return {"u": 0, "r": 1, "d": 2, "l": 3}[name[1]]
    elif name[0] in ("Z", "S", "I"):
        return {"h": 0, "v": 1}[name[1]]
    else: # "O"
        return 0


def piece_max_rotations(name):
    if name[0] in ("T", "J", "L"):
        return 4
    elif name[0] in ("Z", "S", "I"):
        return 2
    else: # "O"
        return 1


def new_piece_name(name, rotations):
    new_raotation_index = (piece_rotation(name) + rotations) % piece_max_rotations(name)
    if name[0] in ("T", "J", "L"):
        return name[0] + "urdl"[new_raotation_index]
    elif name[0] in ("Z", "S", "I"):
        return name[0] + "hv"[new_raotation_index]
    else: # "O"
        return "O"

def new_piece(name, rotations):
    new_name = new_piece_name(name, rotations)
    return tetraminos[new_name]

import copy

def get_move(_grid, piece_name: str, genome):

    # grid = Grid(_grid)
    # shape = Shape(0-2, 5-2, _shape)
    # moves = getAllPossibleMoves(grid, shape, genome)
    # highest_rated_move = max(moves, key = lambda i : i[2])
    # return highest_rated_move
    _grid = Grid(_grid)
    cur_piece = new_piece(piece_name, 0)
    _grid.remove_shape(cur_piece, (-2, 3))
    _grid = _grid.data()

    best_move = (0, 0, -float("inf"))
    for rotations in range(0, 5):
        for translations in range(-5, 6):
            grid = Grid(copy.copy(_grid))

            pos = [-2, 3 + translations] #[row, col]
            piece = new_piece(piece_name, rotations)

            if grid.collides(piece, pos):
                continue

            grid.drop(piece, pos)

            rating = grid.get_rating(genome)

            if rating > best_move[2]:
                best_move = (rotations, translations, rating)

    return best_move

if __name__=="__main__":
    # grid = np.zeros((20, 10), dtype=np.uint8)
    # grid = Grid()
    # shape = Shape(0-2, 5-2, "Tu")

    # # grid.apply_shape(shape)
    # # print(grid)

    # moves = getAllPossibleMoves(grid, shape)

    # highest_rated_move = max(moves, key = lambda i : i[2])
    # print(highest_rated_move)

    # add Tu
    # grid[0, 4] = 255
    # grid[0, 5] = 255
    # grid[0, 6] = 255
    # grid[1, 5] = 255

    # update(grid, "Tu", "Tr")

    # print(grid)

    grid = Grid(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 7, 7, 0, 2, 0, 7, 7, 6, 0],
        [0, 0, 2, 2, 2, 1, 1, 1, 1, 3],
        [4, 4, 5, 0, 2, 5, 5, 3, 3, 3],
        [4, 4, 5, 5, 5, 5, 0, 6, 6, 6],
        [6, 6, 6, 5, 1, 1, 1, 1, 6, 0],
        [2, 6, 7, 7, 0, 0, 0, 3, 4, 4],
    ]))

    print(grid)
    print(grid.get_cumulative_hight())
    print(grid.get_holes())
    print(grid.get_relative_height())
    print(grid.get_roughness())
    print(grid.get_clear_rows())
    print(grid.get_wheight())



    print(new_piece_name("Iv", 0))
    print(new_piece_name("Iv", 1))
    print(new_piece_name("Iv", 2))
    print(new_piece_name("Iv", 3))
    print(new_piece_name("Iv", 4))