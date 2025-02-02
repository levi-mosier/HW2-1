#region imports
import Gauss_Elim as GE
from math import sqrt, pi, exp
#endregion

def GPDF(args):
    """
    GPDF computes the value of the Gaussian (normal) PDF at x for a given mean (mu) and std. dev. (sig).
    The formula is:
        (1 / (sig * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sig)^2)
    :param args: (x, mu, sig)
    :return: float value of the PDF at x
    """
    x, mu, sig = args
    return (1.0 / (sig * sqrt(2.0 * pi))) * exp(-0.5 * ((x - mu) / sig)**2)


def Simpson(fn, args, N=100):
    """
    Implements Simpson's 1/3 rule to numerically integrate fn(x) from x=a to x=b.

    :param fn: function to integrate, called like fn((x, mu, sig)) if needed
    :param args: tuple with (mu, sig, a, b)
    :param N: number of sub-intervals (forced to be even if odd is provided)
    :return: approximate integral of fn from x=a to x=b
    """
    mu, sig, a, b = args

    # ensure N is even
    if N % 2 != 0:
        N += 1

    h = (b - a) / N
    total = fn((a, mu, sig))

    # sum over intermediate points
    for i in range(1, N):
        x_i = a + i * h
        if i % 2 == 0:
            total += 2.0 * fn((x_i, mu, sig))
        else:
            total += 4.0 * fn((x_i, mu, sig))

    # add the last endpoint
    total += fn((b, mu, sig))

    return (h / 3.0) * total


def Probability(PDF, args, c, GT=True):
    """
    Computes P(x > c) or P(x < c) for a normal distribution with mean mu, std dev sig,
    using Simpson's 1/3 rule on the given PDF.

    If GT=False, integrate from [mu - 5*sig, c].
    If GT=True,  integrate from [c, mu + 5*sig].

    :param PDF: callback function for the Gaussian PDF, e.g. GPDF
    :param args: (mu, sig)
    :param c: cutoff value
    :param GT: True => P(x > c), False => P(x < c)
    :return: probability (float)
    """
    mu, sig = args
    left = mu - 5.0 * sig
    right = mu + 5.0 * sig

    if GT:
        # integrate from c to right
        a = max(c, left)   # clamp in case c < left
        b = right
    else:
        # integrate from left to c
        a = left
        b = min(c, right)  # clamp in case c > right

    if b < a:
        return 0.0

    # Use Simpson's rule
    p = Simpson(PDF, (mu, sig, a, b))
    return p


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Uses the Secant method to find a root of fcn(x) = 0 near x0, x1.

    :param fcn: function for which we want a root
    :param x0: initial guess
    :param x1: second guess
    :param maxiter: maximum iterations
    :param xtol: tolerance on consecutive x-values
    :return: (root_estimate, iteration_count)
    """
    f0 = fcn(x0)
    f1 = fcn(x1)

    for i in range(maxiter):
        denom = (f1 - f0)
        if abs(denom) < 1e-15:
            # can't proceed if denominator is ~0
            return (x1, i)
        x2 = x1 - f1 * (x1 - x0) / denom

        if abs(x2 - x1) < xtol:
            return (x2, i + 1)

        # shift
        x0, x1 = x1, x2
        f0, f1 = f1, fcn(x2)

    return (x2, maxiter)


def GaussSeidel(Aaug, x, Niter=15):
    """
    Solves A x = b via Gauss-Seidel iteration on the augmented matrix Aaug = [A|b].
    Ensures diagonal dominance by reordering rows if possible.

    :param Aaug: Nx(N+1) augmented matrix
    :param x: initial guess vector
    :param Niter: number of iterations
    :return: x, the approximate solution
    """
    # Attempt to reorder rows for diagonal dominance
    Aaug = GE.MakeDiagDom(Aaug)

    n = len(Aaug)
    for _ in range(Niter):
        for i in range(n):
            # sum over all j != i
            s = 0.0
            for j in range(n):
                if j != i:
                    s += Aaug[i][j] * x[j]
            x[i] = (Aaug[i][n] - s) / Aaug[i][i]

    return x


def main():
    # Optional internal test
    print("NumericalMethods main() quick tests...")

    # Quick GPDF test
    val = GPDF((0, 0, 1))
    print("GPDF(0|mu=0, sig=1) =", val)

    # Probability test: P(x<0) or P(x>0) for N(0,1)
    p_less = Probability(GPDF, (0,1), 0, GT=False)
    p_greater = Probability(GPDF, (0,1), 0, GT=True)
    print("P(x<0|N(0,1)) =", p_less)
    print("P(x>0|N(0,1)) =", p_greater)

    # Quick Secant example: f(x)=x^2-2 => ~1.414
    def f(x): return x*x - 2
    root, iters = Secant(f, 1, 2)
    print("Root of x^2-2 ~", root, "in", iters, "iterations")

    # Quick GaussSeidel example
    # 2x+y=5, x+3y=9
    # augmented: [[2,1,5],[1,3,9]]
    Aaug_test = [[2.0,1.0,5.0],
                 [1.0,3.0,9.0]]
    guess = [0.0,0.0]
    sol = GaussSeidel(Aaug_test, guess, Niter=15)
    print("GaussSeidel solution for 2x+y=5, x+3y=9 =>", sol)


if __name__ == "__main__":
    main()
