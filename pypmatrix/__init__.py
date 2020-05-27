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

    def trans(self):
        # returns the transpose of the matrix
        m = Matrix(len(self.elements), len(self.elements[0]))
        for (rowindex, row) in enumerate(self.elements):
            for (colindex, el) in enumerate(row):
                m.elements[colindex][rowindex] = el
        return m

el_list = [
[1,2,3],
[4,5,6],
[7,8,9]
]
print(Matrix.from_list(el_list).trans().elements)
