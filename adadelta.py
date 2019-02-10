from inspect import isfunction
from operator import add, truediv, sub, mul


def adadelta(gradient,
             initial_guess, max_iterations=1000,
             decay_parameter=0.5, e=0.000001, tol=0.00001):
    """
        Clean & dependency-free implementation of the ADADELTA
        function minimisation algorithm as presented in
        https://arxiv.org/pdf/1212.5701.pdf

        Note: We interpret lists as vectors.
                  Thus, [1,1,1] is a 3 dimensional vector.

        Arguments:
                REQUIRED

                gradient:
                Return the gradient of the objective function at a
                given vector.
                Takes a n-dim vector and return a n-dim vector of
                partial derivatives.

                initial_guess:
                First guess for a minimum. Used to kick off the algorithm

                ______________________

            OPTIONAL

                max_iterations:
                Number of iterations the algorithm should run.

                decay_parameter:
                Hyperparameter that influences how quickly the
                exponential averages used in the algorithm decay.
                (Choice not significant according to paper)

                e:
                Used to avoid division by 0.
                (Choice not significant according to paper)

                tol:
                norm of gradient that we consider to be "0"

    """
    # Checking inputs

    # gradient must both be a function
    assert isfunction(gradient), "gradient must be a function"

    # initial guess must be a vector
    assert isinstance(initial_guess, list), "initial guess must be a vector"

    sample_gradient = gradient(initial_guess)

    # gradient must output vector
    assert isinstance(
        sample_gradient, list), "gradient must output a vector"

    dim = len(initial_guess)

    # gradient must have the same dimention as parameter space
    assert dim == len(sample_gradient), "Missmatching dimensions"

    # quick implementation of vector operations
    def scalar_mult(vector, scalar):
        # scalar multiplication
        return [scalar * i for i in vector]

    def add_vectors(vector1, vector2):
        # vector addition
        return list(map(add, vector1, vector2))

    def sub_vectors(vector1, vector2):
        # vector subtraction
        return list(map(sub, vector1, vector2))

    def mult_componentwise(vector1, vector2):
        # componentwise vector multiplication
        return list(map(mul, vector1, vector2))

    def divide_componentwise(vector1, vector2):
        # componentwise division of 2 vectors
        return list(map(truediv, vector1, vector2))

    def square(vector):
        # componentwise squaring
        return [x**2 for x in vector]

    def norm(vector):
        # euclidian norm of vector
        return (sum(square(vector)))**0.5

    def RMS(vector):
        # Root mean square function
        # (see paper for details)
        return [(x + e)**0.5 for x in vector]

    p = decay_parameter

    # initialising ADADELTA

    # Exponential average of previous gradients
    E_g = dim * [0]

    # Unit correction parameter
    E_x = dim * [0]

    current_guess = initial_guess

    # Main loop of the iteration
    for count in range(max_iterations):
        # gradient at current position
        g = gradient(current_guess)

        E_g1 = scalar_mult(E_g, p)
        E_g2 = scalar_mult(square(g), 1 - p)

        # updating E_g
        E_g = add_vectors(E_g1, E_g2)

        # Taking root mean squares
        RMS_E_x = RMS(E_x)
        RMS_E_g = RMS(E_g)

        # calculating adaptived learn rate
        learn_rate = divide_componentwise(RMS_E_x, RMS_E_g)

        # minimising step
        delta_x = mult_componentwise(learn_rate, g)

        # updating unit correction
        E_x1 = scalar_mult(E_x, p)
        E_x2 = scalar_mult(square(delta_x), p - 1)
        E_x = add_vectors(E_x1, E_x2)

        # subtracting step
        current_guess = sub_vectors(current_guess, delta_x)

        # break if guess good enough
        if norm(g) < tol:
            break
    return current_guess
