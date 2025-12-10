import numpy as np

# ----------------------------------------------------------
# Initialization of population
# ----------------------------------------------------------
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb


# ----------------------------------------------------------
# Boundary check
# ----------------------------------------------------------
def boundary_check(X, lb, ub):
    X = np.where(X < lb, lb, X)
    X = np.where(X > ub, ub, X)
    return X


# ----------------------------------------------------------
# Gorilla Troops Optimizer (GTO)
# ----------------------------------------------------------
def GTO(pop_size, max_iter, lower_bound, upper_bound, dim, fobj):

    # initialize Silverback
    Silverback = None
    Silverback_Score = np.inf

    # Initialize random population
    X = initialization(pop_size, dim, upper_bound, lower_bound)

    Pop_Fit = np.zeros(pop_size)
    for i in range(pop_size):
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :].copy()

    GX = X.copy()

    lb = np.ones(dim) * lower_bound
    ub = np.ones(dim) * upper_bound

    # controlling parameters (from official paper)
    p = 0.03
    Beta = 3
    w = 0.8

    convergence_curve = np.zeros(max_iter)

    # Main Loop
    for It in range(max_iter):

        # exploration coefficients
        a = (np.cos(2*np.random.rand()) + 1) * (1 - It/max_iter)
        C = a * (2*np.random.rand() - 1)

        # ----------------------------------------------------------
        # Exploration phase
        # ----------------------------------------------------------
        for i in range(pop_size):

            if np.random.rand() < p:
                GX[i, :] = np.random.rand(dim) * (ub - lb) + lb

            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, size=dim)
                    H = Z * X[i, :]
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(pop_size), :] + C * H

                else:
                    r1 = np.random.randint(pop_size)
                    r2 = np.random.randint(pop_size)
                    GX[i, :] = (
                        X[i, :] - C * (
                            C * (X[i, :] - GX[r1, :]) +
                            np.random.rand() * (X[i, :] - GX[r2, :])
                        )
                    )

        GX = boundary_check(GX, lb, ub)

        # Group update
        for i in range(pop_size):
            New_Fit = fobj(GX[i])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i] = GX[i]

            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i].copy()

        # ----------------------------------------------------------
        # Exploitation phase
        # ----------------------------------------------------------
        for i in range(pop_size):

            if a >= w:  # follow the silverback
                g = 2 ** C
                delta = (np.abs(np.mean(GX, axis=0)) ** g) ** (1/g)
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]

            else:  # competition for adult females
                if np.random.rand() >= 0.5:
                    h = np.random.randn(dim)
                else:
                    h = np.random.randn(1)

                r1 = np.random.rand()
                GX[i, :] = (
                    Silverback -
                    (Silverback * (2*r1 - 1) - X[i, :] * (2*r1 - 1)) * (Beta * h)
                )

        GX = boundary_check(GX, lb, ub)

        # Group update again
        for i in range(pop_size):
            New_Fit = fobj(GX[i])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i] = GX[i]

            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i].copy()

        convergence_curve[It] = Silverback_Score
        print(f"Iteration {It+1} | Best Score = {Silverback_Score:.6f}")

    return Silverback_Score, Silverback, convergence_curve


# ----------------------------------------------------------
# Example function (Sphere)
# ----------------------------------------------------------
def sphere(x):
    return np.sum(x**2)


# ----------------------------------------------------------
# Example Run
# ----------------------------------------------------------
if __name__ == "__main__":
    score, best, curve = GTO(
        pop_size=30,
        max_iter=100,
        lower_bound=-10,
        upper_bound=10,
        dim=30,
        fobj=sphere
    )

    print("\nBest Score Found:", score)
    print("Best Position:", best)
