#region imports
from math import sqrt, pi, exp
from NumericalMethods import GPDF, Simpson, Probability
#endregion

def main():
    """
    Demonstrates the Probability(...) function:
      1) Prompt user for mean, stDev, and c.
      2) Ask whether we want P(x>c) or P(x<c).
      3) Compute Probability using our Probability() function from NumericalMethods.
      4) Print the result.

    We can also illustrate the specific homework examples:
      P(x < 105 | N(100, 12.5))
      P(x > 100 + 2*3 | N(100, 3)) => P(x > 106)
    """
    # Example: direct usage (no user input):
    p1 = Probability(GPDF, (100, 12.5), 105, GT=False)
    p2 = Probability(GPDF, (100, 3), 106, GT=True)

    print(f"P(x<105|N(100,12.5))={p1:.4f}")
    print(f"P(x>106|N(100,3))={p2:.4f}")

    print("\nNow, a user-driven example...\n")
    mean_str = input("Population mean? ")
    stdev_str = input("Standard deviation? ")
    c_str = input("c value? ")
    gt_str = input("Probability greater than c? (y/n): ").lower()

    # Convert strings to numerical types
    mean = float(mean_str)
    stDev = float(stdev_str)
    c = float(c_str)
    GT = (gt_str in ["y","yes","true"])

    prob = Probability(GPDF, (mean, stDev), c, GT)
    # Print result
    if GT:
        print(f"P(x>{c}|N({mean},{stDev}))={prob:.6f}")
    else:
        print(f"P(x<{c}|N({mean},{stDev}))={prob:.6f}")

if __name__ == "__main__":
    main()
