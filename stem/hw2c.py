from copy import deepcopy
from NumericalMethods import GaussSeidel

def main():
    """
    Demonstrates the GaussSeidel function by solving two specified systems of linear equations:

    1) A 3x3 system:
         [ [3, 1, -1],
           [1, 4,  1],
           [2, 1,  2] ]
       * [x1, x2, x3]^T
       = [2, 12, 10]^T

       => augmented: [ [3, 1, -1,  2],
                       [1, 4,  1, 12],
                       [2, 1,  2, 10] ]

    2) A 4x4 system:
         [ [ 1, -10,  2,  4],
           [ 3,   1,  4, 12],
           [ 9,   2,  3,  4],
           [-1,   2,  7,  3] ]
       * [x1, x2, x3, x4]^T
       = [2, 12, 21, 37]^T

       => augmented: [ [ 1, -10,  2,  4,  2],
                       [ 3,   1,  4, 12, 12],
                       [ 9,   2,  3,  4, 21],
                       [-1,   2,  7,  3, 37] ]
    """

    # -----------------
    # 1) The 3x3 system
    # -----------------
    Aaug_3x3 = [
        [3.0,  1.0, -1.0,  2.0],
        [1.0,  4.0,  1.0, 12.0],
        [2.0,  1.0,  2.0, 10.0]
    ]
    x_guess_3 = [0.0, 0.0, 0.0]  # initial guess for x1, x2, x3
    sol_3 = GaussSeidel(Aaug_3x3, x_guess_3, Niter=15)
    print("Solution to 3x3 system [3,1,-1; 1,4,1; 2,1,2]:", sol_3)

    # -----------------
    # 2) The 4x4 system
    # -----------------
    Aaug_4x4 = [
        [ 1.0, -10.0,  2.0,  4.0,  2.0],
        [ 3.0,   1.0,  4.0, 12.0, 12.0],
        [ 9.0,   2.0,  3.0,  4.0, 21.0],
        [-1.0,   2.0,  7.0,  3.0, 37.0]
    ]
    x_guess_4 = [0.0, 0.0, 0.0, 0.0]  # initial guess for x1, x2, x3, x4
    sol_4 = GaussSeidel(Aaug_4x4, x_guess_4, Niter=15)
    print("Solution to 4x4 system [1,-10,2,4; 3,1,4,12; 9,2,3,4; -1,2,7,3]:", sol_4)

if __name__ == "__main__":
    main()
