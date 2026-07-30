import numpy as np


def trig_interp(x, y, n_terms):
    """
    Perform trigonometric interpolation using Fourier series.

    Parameters:
    x (array-like): Input values.
    y (array-like): Corresponding function values.
    n_terms (int): Number of Fourier terms to use.

    Returns:
    interpolated_values (array-like): Interpolated function values.
    """

    # Check if input arrays have the same length
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    # Number of data points
    N = len(x)

    # Calculate the coefficients of the Fourier series
    a0 = np.sum(y) / N
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)

    for k in range(1, n_terms + 1):
        an[k - 1] = (2 / N) * np.sum(y * np.cos(2 * np.pi * k * x / (x[-1] - x[0])))
        bn[k - 1] = (2 / N) * np.sum(y * np.sin(2 * np.pi * k * x / (x[-1] - x[0])))

    # Calculate the interpolated values
    interpolated_values = a0 / 2
    for k in range(1, n_terms + 1):
        interpolated_values += an[k - 1] * np.cos(
            2 * np.pi * k * x / (x[-1] - x[0])
        ) + bn[k - 1] * np.sin(2 * np.pi * k * x / (x[-1] - x[0]))

    return interpolated_values, np.array([a0, *an, *bn])


def trig_predict(coeff, x, variance=False):
    """
    Perform trigonometric interpolation using Fourier series.

    Parameters:
    coeff (array-like): Predicted coefficients
    x (array-like): Input values.

    Returns:
    interpolated_values (array-like): Interpolated function values.
    """

    l = len(coeff)
    assert l % 2 == 1, "Odd-expansions only!"

    a0 = coeff[0]
    an = coeff[1 : l // 2 + 1]
    bn = coeff[l // 2 + 1 :]

    n_terms = l // 2

    # Calculate the interpolated values
    if not variance:
        interpolated_values = a0 / 2
        for k in range(1, n_terms + 1):
            interpolated_values += an[k - 1] * np.cos(
                2 * np.pi * k * x / (x[-1] - x[0])
            ) + bn[k - 1] * np.sin(2 * np.pi * k * x / (x[-1] - x[0]))

    else:
        interpolated_values = (a0 / 2) ** 2
        for k in range(1, n_terms + 1):
            interpolated_values += (
                an[k - 1] * np.cos(2 * np.pi * k * x / (x[-1] - x[0])) ** 2
                + bn[k - 1] * np.sin(2 * np.pi * k * x / (x[-1] - x[0])) ** 2
            )

    return interpolated_values


def chebyshev_interp(x, y, n_terms):
    """
    Perform Chebyshev series fitting.

    Parameters:
    x (array-like): Input values.
    y (array-like): Corresponding function values.
    n_terms (int): Number of terms in the Chebyshev series.

    Returns:
    values (array-like): Evaluated values of the Chebyshev series.
    coefficients (array-like): Coefficients of the Chebyshev series.

    """

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    N = len(x)

    x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    cheb_nodes = np.cos((2 * np.arange(1, n_terms + 1) - 1) * np.pi / (2 * n_terms))

    T = np.zeros((N, n_terms))
    for i in range(n_terms):
        T[:, i] = np.cos(i * np.arccos(x_scaled))

    coefficients, _, _, _ = np.linalg.lstsq(T, y, rcond=None)

    n_terms = len(coefficients)
    T = np.zeros((len(x), n_terms))
    for i in range(n_terms):
        T[:, i] = np.cos(i * np.arccos(x_scaled))

    values = T @ coefficients.reshape(-1, 1)

    return values, coefficients


def chebyschev_predict(coefficients, x, variance=False):
    """
    Perform chebyschev interpolation .

    Parameters:
    coefficients (array-like): Predicted coefficients
    x (array-like): Input values.

    Returns:
    values (array-like): Interpolated function values.
    """

    N = len(x)

    x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    n_terms = len(coefficients)
    T = np.zeros((N, n_terms))
    for i in range(n_terms):
        T[:, i] = np.cos(i * np.arccos(x_scaled))

    values = T @ coefficients.reshape(-1, 1)

    if variance:
        values = T**2 @ coefficients.reshape(-1, 1)
    return values


def legendre_interp(x_data, y_data, degree, x_interp=None):
    """
    Performs Legendre interpolation given data points and degree of polynomial.
    """

    def legendre_polynomial(x, n):
        """
        Computes the nth order Legendre polynomial at point x.
        """
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return (
                (2 * n - 1) * x * legendre_polynomial(x, n - 1)
                - (n - 1) * legendre_polynomial(x, n - 2)
            ) / n

    n = degree + 1  # Number of coefficients, including constant term

    x_normalized = 2 * (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) - 1

    V = np.zeros((len(x_normalized), n))
    for i in range(n):
        V[:, i] = legendre_polynomial(x_normalized, i)

    coefficients, _, _, _ = np.linalg.lstsq(V, y_data.reshape(-1, 1))

    if x_interp is None:
        x_interp = x_data.copy()

    x_interp_normalized = (
        2 * (x_interp - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) - 1
    )

    V = np.zeros((len(x_interp_normalized), n))
    for i in range(n):
        V[:, i] = legendre_polynomial(x_interp_normalized, i)

    y_interp = V @ coefficients.reshape(-1, 1)

    return y_interp, coefficients.reshape(1, -1)


def legendre_predict(coefficients, x_interp, variance=False):
    """
    Performs Legendre interpolation given data points and degree of polynomial.
    """

    def legendre_polynomial(x, n):
        """
        Computes the nth order Legendre polynomial at point x.
        """
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return (
                (2 * n - 1) * x * legendre_polynomial(x, n - 1)
                - (n - 1) * legendre_polynomial(x, n - 2)
            ) / n

    n = len(coefficients.squeeze())  # Number of coefficients, including constant term

    x_normalized = (
        2 * (x_interp - np.min(x_interp)) / (np.max(x_interp) - np.min(x_interp)) - 1
    )

    V = np.zeros((len(x_normalized), n))
    for i in range(n):
        V[:, i] = legendre_polynomial(x_normalized, i)

    y_interp = V @ coefficients.reshape(-1, 1)

    if variance:
        y_interp = V**2 @ coefficients.reshape(-1, 1)

    return y_interp.reshape(1, -1)
