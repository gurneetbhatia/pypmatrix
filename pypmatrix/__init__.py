from fractions import Fraction as frac
from itertools import starmap
from operator import mul

class MatrixError(Exception):
    """Base class for other exceptions"""
    def __init__(self, message):
        super().__init__(message)

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

class MatrixNotVectorError(MatrixError):
    """raised when you try carrying out an operation that only works on vectors
    on a non-vector matrix"""
    def __init__(self, message="Matrix needs to be a vector for this operation!"):
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

    def adjoint(self):
        # returns the adjoint of the matrix
        return self.cofactors().trans()

    def cofactor(self, row, col):
        # returns the specified cofactor
        return pow(-1, row + col) * self.minor(row, col)

    def cofactors(self):
        m = self.copy()
        for row, col, element in self:
            m[row][col] = self.cofactor(row, col)
        return m

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
        return self.trans().elements

    def colvec(self, col):
        # returns the specified column as a vector
        el_list = [[row[col]] for row in self.elements]
        return Matrix.from_list(el_list)

    def copy(self):
        return self

    def cross(self, other):
        # find the cross product between two 3D vectors
        if (self.is_vector() and other.is_vector() and self.size() == (3, 1) and other.size() == (3, 1)):
            el_list = [
            [1, 1, 1],
            self.cols_aslist()[0],
            other.cols_aslist()[0]
            ]
            matrix = Matrix.from_list(el_list)
            out_list = [
            [matrix.minor(0, 0)],
            [-1 * matrix.minor(0, 1)],
            [matrix.minor(0, 2)]
            ]
            return Matrix.from_list(out_list)
        else:
            raise MatrixError("Matrices need to be 3D vectors for cross product!")


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
                    # take out the current column (index) from the matrix
                    submatrix = submatrix.del_col(index)
                    determinant += (pow(-1, index) * value * submatrix.det())
                return determinant
        else:
            raise MatrixNotSquareError

    def dir(self):
        # vectors only. returns the unit vector
        if self.is_vector():
            magnitude = self.magnitude()
            # divide each component of the vector by the magnitude
            return self.dot(1/magnitude)
        else:
            raise MatrixNotVectorError

    def dot(self, other):
        # returns the dot product of the object and 'other' if 'other' is matrix
        # otherwise it returns the scalar product assuming it is a real number
        if type(other) == Matrix:
            n1, m1 = self.size()
            n2, m2 = other.size()
            if m1 == n2:
                # a dot product can be computed
                el_list = [[sum(starmap(mul, zip(row, col)))
                for col in zip(*other.elements)] for row in self.elements]
                return Matrix.from_list(el_list)
            else:
                # a dot product cannot be computed
                raise MatrixError("Number of columns in the first matrix need"+
                "to equal to the number of rows in the second matrix!")
        else:
            # work under the assumption that the type is a real Number
            # if the operation is not possible, python will raise the apt err
            el_list = [list(map(lambda el: other*el, row))
            for row in self.elements]
            return Matrix.from_list(el_list)

    def equals(self, other, delta=None):
        # if delta is None, two matrices are equal if they are the same size
        # and their corresponding elements are equal
        # if delta is not None, two matrices are equal if they are the same size
        # and their corresponding elements agree within delta
        if self.size() == other.size():
            if delta is None:
                inequal = False
                for (index, row) in enumerate(self.elements):
                    if row != other.elements[index]:
                        inequal = True
                        break
                return not inequal
            else:
                delta = abs(delta)
                out_of_range = False
                for (i, row) in enumerate(self.elements):
                    for (j, el) in enumerate(row):

                        if (abs(el - other.elements[i][j]) > delta):
                            out_of_range = True
                            break
                    if out_of_range:
                        break
                return not out_of_range
        return False

    def is_square(self):
        # returns true or false to indicate whether a matrix is square or not
        return len(self.elements) == len(self.elements[0])

    def is_vector(self):
        # returns True if the matrix is a vector
        return False not in [len(row) == 1 for row in self.elements]

    def len(self):
        # vectors only. returns the length of the vector
        if self.is_vector():
            return self.size()[0]
        else:
            raise MatrixNotVectorError

    def map(self, func):
        # returns a new matrix formed by mapping func to each element
        matrix = self.copy()
        matrix = [list(map(func, row)) for row in self.elements]
        return matrix

    def magnitude(self):
        # returns the magnitude of a matrix if it is a vector
        if self.is_vector():
            return pow(self.trans().dot(self).elements[0][0], 0.5)
        else:
            raise MatrixNotVectorError

    def minor(self, row, col):
        # returns the specified minor of the matrix
        return self.del_row(row).del_col(col).det()

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
el_list1 = [
[-1,3,5],
[4,4,4],
[9,10,8]
]
matrix3 = Matrix.from_list(el_list1)
# print(matrix.trans().elements)
# print(matrix.det())
# print(matrix.colvec(1).elements)
matrix1 = Matrix.from_list([
[-3],
[4],
[1]
])
matrix2 = Matrix.from_list([
[2],
[1],
[-1]
])

#print(matrix1.dot(matrix2).elements)
print(matrix2.magnitude())
