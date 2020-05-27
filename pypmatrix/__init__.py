from fractions import Fraction as frac


class MatrixError(Exception):
    """Base class for other exceptions"""
    pass

class InequalRowLengthsError(MatrixError):
    """Raised when the row lengths are not all the same"""
    def __init__(self, message="Row lengths in the matrix are not consistent!"):
        super().__init__(message)

class InvalidMatrixElementError(MatrixError):
    """raised when any of the elements in the matrix cannot be processed as
    a fraction object"""
    def __init__(self, message="Matrix contains invalid non-fractional value!"):
        super().__init__(message)

class MatrixNotSquareError(MatrixError):
    """raised when you try carrying out an operation that only works on
    square matrices on a non-square matrix"""
    def __init__(self, message="Matrix not square!"):
        super().__init__(message)

class Matrix(object):
    @staticmethod
    def from_list(el_list):
        m = Matrix(len(el_list), len(el_list[0]))
        row_len = len(el_list[0])
        invalid_el = False
        for (index, row) in enumerate(el_list):
            if len(row) != row_len:
                raise InequalRowLengthsError
            try:
                m.elements[index] = [frac(el) for el in row]
            except ValueError:
                invalid_el = True
                break
        if invalid_el:
            raise InvalidMatrixElementError
        return m

    @staticmethod
    def from_string(string, rowsep=None, colsep=None):
        el_list = [row.split(colsep) for row in string.strip().split(
        rowsep if rowsep != None else '\n'
        )]
        return Matrix.from_list(el_list)

    def __init__(self, rows, cols, fill=0):
        self.elements = [[fill for col in range(cols)] for row in range(rows)]

    def col(self, n):
        # returns an iterator over the specified column
        return iter(self.col_aslist(n))

    def col_aslist(self, n):
        # returns the specified column as a list
        return [row[col] for row in self.elements]

    def cols(self):
        # returns an iterator over each column (which itself is an iterator)
        cols_list = list(map(iter, self.trans().elements))
        return iter(cols_list)

    def cols_aslist(self):
        # returns all the columns as a 2D list
        return self.trans()

    def colvec(self, col):
        # returns the specified column as a vector
        el_list = [[row[col]] for row in self.elements]
        return Matrix.from_list(el_list)

    def copy(self):
        return self

    def del_col(self, col):
        el_list = list(map(
        lambda row: row[:col]+row[col+1:], self.elements
        ))
        return Matrix.from_list(el_list)

    def del_row(self, row):
        el_list = self.elements[:row] + self.elements[row+1:]
        return Matrix.from_list(el_list)

    def det(self):
        # returns the determinant of the matrix if it is square
        if self.is_square():
            if len(self.elements) == 1:
                return self.elements[0][0]
            # now find and return the determinant of the matrix
            if self.size() == (2, 2):
                return ((self.elements[0][0]*self.elements[1][1]) -
                (self.elements[0][1]*self.elements[1][0]))
            else:
                determinant = 0
                for (index, value) in enumerate(self.elements[0]):
                    # iterate through the top row of the matrix
                    # take out the top row from the matrix copy
                    submatrix = self.del_row(0)
                    # submatrix = self.elements[1:]
                    # take out the current column (index) from the matrix
                    #submatrix = list(map(
                    #lambda row: row[:index]+row[index+1:], submatrix))
                    submatrix = submatrix.del_col(index)
                    determinant += (pow(-1, index) * value * submatrix.det())
                return determinant
        else:
            raise MatrixNotSquareError

    def is_square(self):
        # returns true or false to indicate whether a matrix is square or not
        return len(self.elements) == len(self.elements[0])

    def size(self):
        # returns a tuple with number of rows followed by number of columns
        return (len(self.elements), len(self.elements[0]))

    def trans(self):
        # returns the transpose of the matrix
        m = Matrix(len(self.elements[0]), len(self.elements))
        for (rowindex, row) in enumerate(self.elements):
            for (colindex, el) in enumerate(row):
                m.elements[colindex][rowindex] = el
        return m

el_list = [
[1,2,6],
[4,5,6],
[7,8,9]
]
matrix = Matrix.from_list(el_list)
# print(matrix.trans().elements)
# print(matrix.det())
print(matrix.colvec(1).elements)
