# region explanation
# This program is used to teach Gauss elimination by row operations
# The elementary operations for Gauss elimination are:
#  1. Swap the positions of two rows
#  2. Multiply a row by a non-zero scalar
#  3. Add to one row, a scalar multiple of another row
# endregion

# region imports
import copy as CP
from copy import deepcopy as dc  # a quick way to access deepcopy
# endregion

def FirstNonZero_Index(R):
    """
    Finds the pivot for a row (the first non-zero number from left to right).
    :param R: a row vector
    :return: the column index of the first non-zero number, or -1 if no non-zero found
    """
    for ColumnIndex in range(len(R)):
        if R[ColumnIndex] != 0.0:
            return ColumnIndex
    return -1

def SwapRows(A, r1, r2):
    """
    Elementary row operation: swap row r1 with row r2 in matrix A.
    :param A: matrix
    :param r1: row index
    :param r2: row index
    :return: A with swapped rows
    """
    rmax = max(r1, r2)
    rmin = min(r1, r2)
    RMax = A[rmax]
    RMin = A.pop(rmin)  # pop row rmin
    A.insert(rmin, RMax)  # insert old RMax at position rmin
    A[rmax] = RMin
    return A

def MultRow(R, s=1):
    """
    Multiply row vector R by scalar s.
    :param R: the row vector
    :param s: the scalar
    :return: new row vector with each element multiplied by s
    """
    for i in range(len(R)):
        R[i] *= s
    return R

def AddRows(R1, R2, s=1.0):
    """
    R1 + s*R2 (element-wise).
    :param R1: row vector
    :param R2: row vector
    :param s: scalar
    :return: new row vector
    """
    RNew = CP.deepcopy(R1)
    for i in range(len(R1)):
        RNew[i] += R2[i] * s
    return RNew

def MakeDiagDom(A):
    """
    Reorders rows of matrix A so that the diagonal element in each row is (hopefully)
    the largest absolute value in that row. This is a common step to improve stability
    for Gauss-Seidel or Gauss Elimination.

    :param A: Nx(N+1) matrix if augmented, or NxN if just coefficient matrix
    :return: the same matrix A, but row-reordered in-place
    """
    # We'll do a simple partial pivot approach for each row i:
    n = len(A)
    for i in range(n):
        # Find the row pivotRow in [i, n-1] where |A[pivotRow][i]| is largest
        pivotRow = i
        maxVal = abs(A[i][i])
        for r in range(i+1, n):
            if abs(A[r][i]) > maxVal:
                pivotRow = r
                maxVal = abs(A[r][i])
        if pivotRow != i:
            A = SwapRows(A, i, pivotRow)
    return A

def EchelonForm(A):
    """
    Perform row operations (Gauss elimination) to produce an upper triangular (echelon) form of A.
    :param A: matrix
    :return: the echelon form of A (upper triangular)
    """
    m = len(A)
    n = len(A[0])
    Ech = CP.deepcopy(A)

    # for each row i
    for i in range(m):
        # pivot row search
        for r in range(i, m):
            p = FirstNonZero_Index(Ech[r])
            if p == i:
                Ech = SwapRows(Ech, r, i)
                break
        # eliminate below pivot
        if Ech[i][i] != 0.0:
            for r in range(i+1, m):
                p = FirstNonZero_Index(Ech[r])
                if p == i:
                    Row = Ech[r]
                    s = -Ech[r][p] / Ech[i][i]
                    Ech[r] = AddRows(Row, Ech[i], s)
    return Ech

def ReducedEchelonForm(A):
    """
    Creates a Reduced Echelon Form (RREF) from A by:
      1) Echelon form
      2) Normalize pivots to 1
      3) Eliminate above pivots
    :param A: matrix
    :return: the RREF of A
    """
    REF = EchelonForm(A)
    m = len(A)
    for i in range(m-1, -1, -1):
        R = REF[i]
        j = FirstNonZero_Index(R)
        if j == -1:
            continue  # entire row is zero
        # normalize pivot to 1
        pivotVal = R[j]
        if pivotVal != 0:
            R = MultRow(R, 1.0/pivotVal)
        REF[i] = R
        # eliminate above
        for ii in range(i-1, -1, -1):
            if REF[ii][j] != 0:
                REF[ii] = AddRows(REF[ii], R, -REF[ii][j])
    return REF

def IDMatrix(A):
    """
    Produce an identity matrix of the same row/column size as A.
    :param A: NxN matrix
    :return: identity NxN
    """
    m = len(A)
    n = len(A[0])
    # create identity of dimension m x n if square
    IM = [[0]*n for _ in range(m)]
    for i in range(min(m,n)):
        IM[i][i] = 1
    return IM

def AugmentMatrix(A, B):
    """
    Augment matrix A with matrix B by concatenating the rows horizontally.
    :param A: NxM
    :param B: NxP
    :return: Nx(M+P)
    """
    C = CP.deepcopy(A)
    for i in range(len(C)):
        C[i] += B[i]
    return C

def popColumn(A, j):
    """
    Remove column j from matrix A.
    :param A: matrix
    :param j: column index
    :return: (column_vector, matrix_with_column_removed)
    """
    AA = dc(A)
    col = []
    for rowIndex in range(len(AA)):
        col.append(AA[rowIndex].pop(j))
    return col, AA

def insertColumn(A, b, i):
    """
    Insert column vector b into matrix A at index i, pushing others to the right.
    :param A: matrix
    :param b: column vector
    :param i: insertion index
    :return: new matrix
    """
    ANew = dc(A)
    for r in range(len(ANew)):
        ANew[r].insert(i, b[r])
    return ANew

def replaceColumn(A, b, i):
    """
    Replace column i of A with column vector b.
    :param A: matrix
    :param b: column vector
    :param i: column index
    :return: new matrix
    """
    ANew = dc(A)
    # first pop column i
    _, temp = popColumn(ANew, i)
    # then insert new column b
    out = insertColumn(temp, b, i)
    return out

def InvertMatrix(A):
    """
    Finds the inverse of NxN matrix A by forming [A|I], then row-reducing to [I|A_inv].
    :param A: NxN matrix
    :return: A_inv, the inverse of A
    """
    ID = IDMatrix(A)
    Ainv = AugmentMatrix(A, ID)
    IAinv = ReducedEchelonForm(Ainv)
    # the inverse is the right half
    nCols = len(A[0])
    for j in range(nCols-1, -1, -1):
        # remove the left half columns
        _ , IAinv = popColumn(IAinv, j)
    return IAinv

def MatrixMultiply(A, B):
    """
    Multiply (m x n) matrix A with (n x p) matrix B => (m x p).
    :param A: m x n
    :param B: n x p
    :return: m x p matrix
    """
    m = len(A)
    n = len(A[0])
    nn = len(B)
    p = len(B[0])
    if n != nn:
        raise ValueError("Incompatible dimensions for matrix multiply.")
    C = [[0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]
    return C

def main():
    # Example usage of some of these functions:
    M = [[4, -1, -1, 3],
         [-2, -3, 1, 9],
         [-1, 1, 7, -6]]
    print("Original matrix:")
    for r in M:
        print(r)

    E = EchelonForm(M)
    print("Echelon form:")
    for r in E:
        print(r)

    RREF = ReducedEchelonForm(M)
    print("Reduced Echelon Form:")
    for r in RREF:
        print(r)

if __name__ == "__main__":
    main()
