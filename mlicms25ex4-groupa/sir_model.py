import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def mu(b, I, mu0, mu1):
    """
    Computes the recovery rate mu; computed with the parameters b, mu0, mu1 and I.

    Arguments:
    ----------
    b : float
        Number of beds per 10,000 people.
    I : float
        Number of infected people.
    mu0: float
        Minimum recovery rate based on the number of available beds.
    mu1: float
        Maximum recovery rate based on the number of available beds.

    Returns:
    --------
    mu : float
        Recovery rate that depends on mu0, mu1, b, and I.
    """
    mu = mu0 + (mu1 - mu0) * (
        b / (I + b)
    )  # Calculate recovery rate; depends on mu0, mu1, b
    return mu  # Return recovery rate


def R0(beta, d, nu, mu1):
    """
    Computes the basic reproduction number; computed with the parameters beta, d, nu and mu1.

    Arguments:
    ----------
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    d : float
        Per capita natural death rate.
    nu : float
        Per capita disease-induced death rate.
    mu1 : float
        Maximum recovery rate based on the number of available beds.

    Returns:
    r0 : float
        Basic reproduction that depends on beta, d, nu, and mu1.
    --------

    """
    r0 = beta / (
        d + nu + mu1
    )  # Calculate basic reproduction rate; depends on beta, d, nu, and mu1
    return r0  # Return basic reproduction rate


def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcation analysis used to study the epidemic model.

    Arguments:
    ----------
    I : float
        Number of infected people.
    mu0: float
        Minimum recovery rate based on the number of available beds.
    mu1: float
        Maximum recovery rate based on the number of available beds.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate (or birth rate) of susceptible population.
    d : float
        Per capita natural death rate.
    nu : float
        Per capita disease-induced death rate.
    b : float
        Number of beds per 10,000 people.

    Returns:
    res : float
        Value of the indicator function.
    --------
    """
    # Calculate the indicator function value that depends on I, mu0, mu1, beta, A, d, nu, and b
    c0 = b**2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res  # Return the indicator function value


def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    Continuous SIR-type epidemic model with hospitalization and natural/disease-induced death.

    Arguments:
    ----------
    y : List[float, float, float]
        S, I, and R values of the SIR-type model given in a list.
    mu0: float
        Minimum recovery rate based on the number of available beds.
    mu1: float
        Maximum recovery rate based on the number of available beds.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate (or birth rate) of susceptible population.
    d : float
        Per capita natural death rate.
    nu : float
        Per capita disease-induced death rate.
    b : float
        Number of beds per 10,000 people.

    Returns:
    List[float, float, float]
        List of the change of susceptible (S), infective (I), and removed (R) people in the continuous model.
    --------
    """
    S, I, R = y[:]  # Get the S, I, and R values of the model
    m = mu(b, I, mu0, mu1)  # Compute the recovery rate

    dSdt = (
        A - (d * S) - ((beta * S * I) / (S + I + R))
    )  # Compute the change of susceptiple (S) people in the continuous model
    dIdt = (
        (-(d + nu) * I) - (m * I) + ((beta * S * I) / (S + I + R))
    )  # Compute the change of infective (I) people in the continuous model
    dRdt = (m * I) - (
        d * R
    )  # Compute the change of removed (R) people in the continuous model

    return [dSdt, dIdt, dRdt]  # Return the change of S, I, and R values of the model


def plot_continuous_SIR(sol, mu0, mu1, beta, A, d, nu, b):
    """
    Plots 3 graphs:
        - S, I, and R variables over time
        - Recovery rate and infective variable over time
        - Indicator function value given infected people

    Arguments:
    ----------
    sol : scipy.integrate.OdeResult
        Solution to the ODE system over the time grid.
    mu0: float
        Minimum recovery rate based on the number of available beds.
    mu1: float
        Maximum recovery rate based on the number of available beds.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate (or birth rate) of susceptible population.
    d : float
        Per capita natural death rate.
    nu : float
        Per capita disease-induced death rate.
    b : float
        Number of beds per 10,000 people.

    Returns:
        None
    --------
    """

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label="1E0*susceptible")
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label="1E3*infective")
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label="1E1*removed")
    ax[0].set_xlim([0, 500])
    ax[0].legend()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label="recovery rate")
    ax[1].plot(sol.t, 1e2 * sol.y[1], label="1E2*infective")
    ax[1].set_xlim([0, 500])
    ax[1].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    I_h = np.linspace(-0.0, 0.05, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b))
    ax[2].plot(I_h, 0 * I_h, "r:")
    # ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")

    fig.tight_layout()


def plot_trajectories(t_0, mu0, mu1, beta, A, d, nu, b, rtol, atol):
    """
    Plots trajectories for three initial starting points given.

    Arguments:
    ----------
    t_0 : float
        Starting time.
    mu0: float
        Minimum recovery rate based on the number of available beds.
    mu1: float
        Maximum recovery rate based on the number of available beds.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate (or birth rate) of susceptible population.
    d : float
        Per capita natural death rate.
    nu : float
        Per capita disease-induced death rate.
    b : float
        Number of beds per 10,000 people.
    rtol: float
        Error tolerance to make the solution qualitatively correct.
    atol: float
        Error tolerance to make the solution qualitatively correct.

    Returns:
        None
    --------
    """

    NT = 15000
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    time = np.linspace(t_0, 15000, NT)

    # cmap = ["BuPu", "Purples", "bwr"][1]

    SIM0 = [
        195.3,
        0.052,
        4.4,
    ]  # what happens with this initial condition when b=0.022? -- it progresses VERY slowly. Needs t_end to be super large.
    sol = solve_ivp(
        model,
        t_span=[time[0], time[-1]],
        y0=SIM0,
        t_eval=time,
        args=(mu0, mu1, beta, A, d, nu, b),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    ax.plot(sol.y[0], sol.y[1], sol.y[2], "r-")
    # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap='bwr')

    SIM0 = [195.7, 0.03, 3.92]  # what happens with this initial condition when b=0.022?
    sol = solve_ivp(
        model,
        t_span=[time[0], time[-1]],
        y0=SIM0,
        t_eval=time,
        args=(mu0, mu1, beta, A, d, nu, b),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    ax.plot(sol.y[0], sol.y[1], sol.y[2], "g-")
    # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap)

    SIM0 = [193, 0.08, 6.21]  # what happens with this initial condition when b=0.022?
    sol = solve_ivp(
        model,
        t_span=[time[0], time[-1]],
        y0=SIM0,
        t_eval=time,
        args=(mu0, mu1, beta, A, d, nu, b),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    ax.plot(sol.y[0], sol.y[1], sol.y[2], "b-")
    # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap)

    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")

    ax.set_title(f"SIR trajectory, beta = {beta}")
    fig.tight_layout()

    def plot_trajectories_1(t_0, mu0, mu1, beta, A, d, nu, b, rtol, atol):
        """
        Plots trajectories for three initial starting points given.

        Arguments:
        ----------
        t_0 : float
            Starting time.
        mu0: float
            Minimum recovery rate based on the number of available beds.
        mu1: float
            Maximum recovery rate based on the number of available beds.
        beta : float
            Average number of adequate contacts per unit time with infectious individuals.
        A : float
            Recruitment rate (or birth rate) of susceptible population.
        d : float
            Per capita natural death rate.
        nu : float
            Per capita disease-induced death rate.
        b : float
            Number of beds per 10,000 people.
        rtol: float
            Error tolerance to make the solution qualitatively correct.
        atol: float
            Error tolerance to make the solution qualitatively correct.

        Returns:
            None
        --------
        """

        NT = 15000
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        time = np.linspace(t_0, 15000, NT)

        # cmap = ["BuPu", "Purples", "bwr"][1]

        SIM0 = [
            195.3,
            0.052,
            4.4,
        ]  # what happens with this initial condition when b=0.022? -- it progresses VERY slowly. Needs t_end to be super large.
        sol = solve_ivp(
            model,
            t_span=[time[0], time[-1]],
            y0=SIM0,
            t_eval=time,
            args=(mu0, mu1, beta, A, d, nu, b),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        ax.plot(sol.y[0], sol.y[1], sol.y[2], "r-")
        # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap='bwr')

        SIM0 = [195.7, 0.03, 3.92]  # what happens with this initial condition when b=0.022?
        sol = solve_ivp(
            model,
            t_span=[time[0], time[-1]],
            y0=SIM0,
            t_eval=time,
            args=(mu0, mu1, beta, A, d, nu, b),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        ax.plot(sol.y[0], sol.y[1], sol.y[2], "g-")
        # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap)

        SIM0 = [193, 0.08, 6.21]  # what happens with this initial condition when b=0.022?
        sol = solve_ivp(
            model,
            t_span=[time[0], time[-1]],
            y0=SIM0,
            t_eval=time,
            args=(mu0, mu1, beta, A, d, nu, b),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        ax.plot(sol.y[0], sol.y[1], sol.y[2], "b-")
        # ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap)

        ax.set_xlabel("S")
        ax.set_ylabel("I")
        ax.set_zlabel("R")

        ax.set_title(f"SIR trajectory, b = {b}")
        fig.tight_layout()
