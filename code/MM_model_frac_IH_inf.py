"""
Author:            Eva Nieuwenhuis
University group:  Biosystems Data Analysis Group, UvA
Student ID:        13717405

Description:  Code with the model that simulates the dynamics in the multiple
              myeloma (MM) microenvironment with four cell types: drug-sensitive
              MM cells (MMd), resistant MM cells (MMr), osteoblasts (OB) and
              osteoclasts (OC). The model is a public goods game in the framework
              of evolutionary game theory with collective interactions and linear
              benefits. In this model, there is looked at the fractions of the
              four cell types. The IHs have not only an influence on the MMd but
              also on the OB and OC.

Example interaction matrix:
M = np.array([
       Foc Fob Fmmd Fmmr
    OC  [a,  b,  c,  d],
    OB  [e,  f,  g,  h],
    MMd [i,  j,  k,  l],
    MMr [m,  n,  o,  p]])
"""

# Import the needed libraries
import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import doctest

def main():
    # Do doc tests
    doctest.testmod()

    # Make a figure showing the cell fraction dynamics by traditional therapy and
    # by adaptive therapy
    list_t_steps_drug = [5, 5, 5]
    Figure_continuous_MTD_vs_AT_s_and_w_a_h(18, list_t_steps_drug)

    # Make a 3D figure showthing the effect of different drug holiday and
    # administration periods
    Figure_3D_MM_frac_IH_add_and_holiday()


def fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of an osteoclast in a population.

    Parameters:
    -----------
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMd: Float
        Fraction of the MMd.
    xMMr: Float
        Fraction of the MMr.
    N: Int
        Fraction of individuals within the interaction range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOC: Float
        Fitness of an OC.

    Example:
    -----------
    >>> fitness_WOC(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    0.10859999999999997
    """
    # Extract the needed matrix values
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[0, 3]

    # Calculate the fitness value
    WOC = (a*xOC*cOC + b*xOB*cOB + c*xMMd*cMMd + d* xMMr *cMMr)*(N - 1)/N - cOC
    return WOC

def fitness_WOB(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of an osteoblast in a population.

    Parameters:
    -----------
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMd: Float
        Fraction of the MMd.
    xMMr: Float
        Fraction of the MMr.
    N: Int
        Fraction of individuals within the interaction range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WOB: Float
        Fitness of an OB.

    Example:
    -----------
    >>> fitness_WOB(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    -0.020900000000000002
    """
    # Extract the necessary matrix values
    e = matrix[1, 0]
    f = matrix[1, 1]
    g = matrix[1, 2]
    h = matrix[1, 3]

    # Calculate the fitness value
    WOB = (e*xOC*cOC + f*xOB*cOB + g*xMMd*cMMd + h* xMMr*cMMr)*(N - 1)/N - cOB
    return WOB

def fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix,
                                                            WMMd_inhibitor = 0):
    """
    Function that calculates the fitness of a drug-senstive MM cell in a
    population.

    Parameters:
    -----------
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMd: Float
        Fraction of the MMd.
    xMMr: Float
        Fraction of the MMr.
    N: Int
        Fraction of individuals within the interaction range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    WMMd: Float
        Fitness of a MMd.

    Example:
    -----------
    >>> fitness_WMMd(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]), 0)
    0.05730000000000007
    """
    # Extract the necessary matrix values
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]
    l = matrix[2, 3]

    # Calculate the fitness value
    WMMd = (i*xOC*cOC + j*xOB*cOB + k*xMMd*cMMd + l* xMMr*cMMr - WMMd_inhibitor\
                                                            )*(N - 1)/N - cMMd
    return WMMd

def fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix):
    """
    Function that calculates the fitness of a resistant MM cell in a population.

    Parameters:
    -----------
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMd: Float
        Fraction of the MMd.
    xMMr: Float
        Fraction of the MMr.
    N: Int
        Fraction of individuals within the interaction range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    WMMr: Float
        Fitness of a MMr.

    Example:
    -----------
    >>> fitness_WMMr(0.4, 0.2, 0.3, 0.1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    -0.23539999999999994
    """
    # Extract the necessary matrix values
    m = matrix[3, 0]
    n = matrix[3, 1]
    o = matrix[3, 2]
    p = matrix[3, 3]

    # Calculate the fitness value
    WMMr = (m*xOC*cOC + n*xOB*cOB + o*xMMd*cMMd + p* xMMr*cMMr)*(N - 1)/N - cMMr
    return WMMr

def model_dynamics(y, t, N, cOC, cOB, cMMd, cMMr, matrix, WMMd_inhibitor = 0):
    """Function that determines the fracuenty dynamics in a population over time.

    Parameters:
    -----------
    y: List
        List with the values of xOC, xOB, xMMd and xMMr.
    t: Numpy.ndarray
        Array with all the time points.
    N: Int
        Fraction of cells in the difussion range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMd: Float
        Cost parameter MMd.
    cMMr: Float
        Cost parameter MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    [xOC_change, xOB_change, xMMd_change, xMMr_change]: List
        List containing the changes in fractions of OC, OB, MMd and MMr.

    Example:
    -----------
    >>> model_dynamics([0.4, 0.2, 0.3, 0.1], 1, 10, 0.3, 0.2, 0.3, 0.5, np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    [0.030275999999999983, -0.010762000000000006, 0.0073170000000000145, -0.026830999999999994]
    """
    xOC, xOB, xMMd, xMMr = y

    # Determine the fitness values
    WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
    WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
    WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix,
                                                                WMMd_inhibitor)
    WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)

    # Determine the average fitness
    W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

    # Determine the new fractions based on replicator dynamics
    xOC_change = xOC * (WOC - W_average)
    xOB_change = xOB * (WOB - W_average)
    xMMd_change = xMMd * (WMMd - W_average)
    xMMr_change = xMMr * (WMMr - W_average)

    # Make floats of the arrays
    xOC_change = float(xOC_change)
    xOB_change = float(xOB_change)
    xMMd_change = float(xMMd_change)
    xMMr_change = float(xMMr_change)

    return [xOC_change, xOB_change, xMMd_change, xMMr_change]


def frac_to_fitness_values(dataframe_fractions, N, cOC, cOB, cMMd, cMMr, matrix,
                                                            WMMd_inhibitor = 0):
    """Function that determines the fitness values of the OC, OB, MMd and MMr
    based on their fractions on every time point. It also calculates the average
    fitness.

    Parameters:
    -----------
    dataframe_fractions: Dataframe
        Dataframe with the fractions of the OB, OC MMd and MMr on every
        timepoint.
    N: Int
        Fraction of cells in the difussion range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    dataframe_fitness: Dataframe
        A dataframe with the fitness values of the OB, OC, MMd and MMr and
        the average fitness on every time point.
    """
    # Make lists
    WOC_list = []
    WOB_list = []
    WMMd_list = []
    WMMr_list = []
    W_average_list = []
    generation_list = []

    # Iterate over each row
    for index, row in dataframe_fractions.iterrows():

        # Extract values of xOC, xOB, and xMM for the current row
        xOC = row['xOC']
        xOB = row['xOB']
        xMMd = row['xMMd']
        xMMr = row['xMMr']

        # Determine the fitness values
        WOC = fitness_WOC(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, matrix)
        WOB = fitness_WOB(xOC, xOB, xMMd, xMMr,  N, cOC, cOB, cMMd, cMMr, matrix)
        WMMd = fitness_WMMd(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                                        matrix, WMMd_inhibitor)
        WMMr = fitness_WMMr(xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                                                                        matrix)

        # Determine the average fitness
        W_average = xOC * WOC + xOB * WOB + xMMd * WMMd + xMMr * WMMr

        # Add the calculated fitness values to the respective lists
        WOC_list.append(WOC)
        WOB_list.append(WOB)
        WMMd_list.append(WMMd)
        WMMr_list.append(WMMr)
        W_average_list.append(W_average)
        generation_list.append(index)

    # Create a datafrane with the calculated fitness values
    dataframe_fitness = pd.DataFrame({'Generation': generation_list,
                            'WOC': WOC_list, 'WOB': WOB_list, 'WMMd': WMMd_list,
                                'WMMr': WMMr_list, 'W_average': W_average_list})

    return(dataframe_fitness)

def combine_dataframes(df_1, df_2):
    """ Function that combines two datafranes in on dataframe

    Parameters:
    -----------
    df_1: DataFrame
        The first dataframe containing the collected data.
    df_2: DataFrame
        The second dataframe containing the collected data.

    Returns:
    --------
    combined_df: DataFrame
        Dataframe that is a combination of the two dataframes
    """
    # Check if the dataframes are empty
    if df_1.empty or df_2.empty:
        # return the dataframe that is not empty
        combined_df = df_1 if not df_1.empty else df_2

    else:
        # delete the NA columns
        df_1 = df_1.dropna(axis=1, how='all')
        df_2 = df_2.dropna(axis=1, how='all')

        # Combine the dataframes
        combined_df = pd.concat([df_1, df_2], ignore_index=True)

    return(combined_df)

def save_dataframe(data_frame, file_name, folder_path):
    """ Function that saves a dataframe as csv file.

    Parameters:
    -----------
    data_frame: DataFrame
        The dataframe containing the collected data.
    file_name: String
        The name of the csv file.
    folder_path: String
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    data_frame.to_csv(file_path, index=False)

def save_Figure(Figure, file_name, folder_path):
    """Save the Figure to a specific folder.

    Parameters:
    -----------
    Figure: Matplotlib Figure
        Figure object that needs to be saved.
    file_name: String
        The name for the plot.
    folder_path: String:
        Path to the folder where the data will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    Figure.savefig(os.path.join(folder_path, file_name))

def switch_dataframe(time_start_drugs, n_switches, t_steps_drug, t_steps_no_drug,
                xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, cOC_IH, cOB_IH,
                matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values
    over time for a given time of drug holiday and administration periods.

    Parameters:
    -----------
    time_start_drugs: Int
        The generation after which the inhibitors will be administared.
    n_switches: Int
        The fraction of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The fraction of generations drugs are administared.
    t_steps_no_drug: Int
        The fraction of generations drugs are not administared.
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMr: Float
        Fraction of the MMr.
    xMMd: Float
        Fraction of the MMd.
    N: Int
        Fraction of cells in the difussion range.
    cOC: Float
        Cost parameter OC.
    cOB: Float
        Cost parameter OB.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    cOC_IH: Float
        Cost parameter OC when a IH is administered.
    cOB_IH: Float
        Cost parameter OB when a IH is administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the xOC, xOB, xMMd and xMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = time_start_drugs
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'xOC':y[:, 0], 'xOB':y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a fraction of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:

            # Determine the start fraction values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_GF_IH

            t = np.linspace(time, time + t_steps_drug, t_steps_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC_IH, cOB_IH, cMMd, cMMr, matrix, WMMd_inhibitor)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 1
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Determine the start fraction values
            xOC = df_total_switch['xOC'].iloc[-1]
            xOB = df_total_switch['xOB'].iloc[-1]
            xMMd = df_total_switch['xMMd'].iloc[-1]
            xMMr = df_total_switch['xMMr'].iloc[-1]

            # Payoff matrix
            matrix = matrix_no_GF_IH

            t = np.linspace(time, time + t_steps_no_drug , t_steps_no_drug)
            y0 = [xOC, xOB, xMMd, xMMr]
            parameters = (N, cOC, cOB, cMMd, cMMr, matrix)

            # Determine the ODE solutions
            y = odeint(model_dynamics, y0, t, args=parameters)
            df = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
                'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

            # Add dataframe to total dataframe
            df_total_switch = combine_dataframes(df_total_switch, df)
            df_total_switch.reset_index(drop=True, inplace=True)

            # Change the x and time value
            x = 0
            time += t_steps_no_drug

    return df_total_switch

def continuous_add_IH_df(time_start_drugs, end_generation, xOC, xOB, xMMd, xMMr,
        N, cOC, cOB, cMMd, cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_GF_IH,
        WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the cell type fractions when the IHs
    are administered continuously.

    Parameters:
    -----------
    time_start_drugs: Int
        The generation after which the inhibitors will get administared
    end_generation: Int
        The last generation for which the fractions have to be calculated
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMd: Float
        Fraction of the MMd.
    xMMr: Float
        Fraction of the MMr.
    N: Int
        Fraction of cells in the difussion range.
    cOC: Float
        Cost parameter OC.
    cOB: float
        Cost parameter OB.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    cOC_IH: Float
        Cost parameter OC when a IH is administered.
    cOB_IH: Float
        Cost parameter OB when a IH is administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                administrated.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administrated.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total: DataFrame
        The dataframe with cell fractions when IHs are continiously administered.
    """
    t = np.linspace(0, time_start_drugs, time_start_drugs)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC, cOB, cMMd, cMMr, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
            'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

    # Determine the current fractions
    xOC = df_1['xOC'].iloc[-1]
    xOB = df_1['xOB'].iloc[-1]
    xMMd = df_1['xMMd'].iloc[-1]
    xMMr = df_1['xMMr'].iloc[-1]

    t = np.linspace(time_start_drugs, end_generation, 120)
    y0 = [xOC, xOB, xMMd, xMMr]
    parameters = (N, cOC_IH, cOB_IH, cMMd, cMMr, matrix_GF_IH, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'xOC': y[:, 0], 'xOB': y[:, 1],
            'xMMd': y[:, 2], 'xMMr': y[:, 3], 'total xMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = combine_dataframes(df_1, df_2)

    return df_total

def minimal_tumour_frac_t_steps(t_steps_drug, t_steps_no_drug, xOC, xOB, xMMd,
                            xMMr, N, cOC, cOB, cMMd, cMMr, cOC_IH, cOB_IH,
                            matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the xOC, xOB, xMMd and xMMr values
    over time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The fraction of generations drugs are administared.
    t_steps_no_drug: Int
        The fraction of generations drugs are not administared.
    xOC: Float
        Fraction of OC.
    xOB: Float
        Fraction of OB.
    xMMr: Float
        Fraction of the MMr.
    xMMd: Float
        Fraction of the MMd.
    N: Int
        fraction of cells in the difussion range.
    cOC: Float
        Cost parameter OC.
    cOB: float
        Cost parameter OB.
    cMMr: Float
        Cost parameter MMr.
    cMMd: Float
        Cost parameter MMd.
    cOC_IH: Float
        Cost parameter OC when a IH is administered.
    cOB_IH: Float
        Cost parameter OB when a IH is administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
                                                                administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_fraction: Float
        The average MM fraction in the equilibrium.

    Example:
    -----------
    >>> matrix_no_GF_IH = np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]])
    >>> matrix_no_GF_IH = np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [0.8, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]])
    >>> minimal_tumour_frac_t_steps(5, 5, 0.2, 0.3, 0.2, 0.3, 10, 0.3, 0.2, 0.3,
    ...                         0.5, 0.4, 0.2, matrix_no_GF_IH, matrix_no_GF_IH)
    4.533166876036014e-11
    """
    # Deteremine the fraction of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((110 // time_step) -1)

    # Create a dataframe of the fractions
    df = switch_dataframe(15, n_switches, t_steps_drug, t_steps_no_drug, xOC,
                    xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr, cOC_IH, cOB_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM fraction in the last period with and without drugs
    last_MM_fractions = df['total xMM'].tail(int(time_step *2))
    average_MM_fraction = last_MM_fractions.sum() / (int(time_step*2))

    return float(average_MM_fraction)

def x_y_z_axis_values_3d_plot(dataframe, name):
    """ Function that determines the x, y and z axis values from the given
    dataframe. It also prints the administration and holiday duration leading
    to the lowest total MM fraction in the equilibrium

    Parameters:
    -----------
    Dataframe: dataFrame
        The dataframe with the generated data
    name: String
        The name of the administered IH(s)

    Returns:
    --------
    X_values: Numpy.ndarray
        Array with the values for the x-axis
    Y_values: Numpy.ndarray
        Array with the values for the y-axis
    Z_values: Numpy.ndarray
        Array with the values for the z-axis
    """

    # Find the drug administration and holiday period causing the lowest MM
    # fraction
    min_index =  dataframe['MM fraction'].idxmin()
    g_no_drug_min = dataframe.loc[min_index, 'Generations no drug']
    g_drug_min = dataframe.loc[min_index, 'Generations drug']
    frac_min = dataframe.loc[min_index, 'MM fraction']

    print(f"""Lowest MM fraction: {frac_min}-> MMd {name} holidays are
            {g_no_drug_min} generations and MMd {name} administrations
            are {g_drug_min} generations""")

    # Avoid errors because of the wrong datatype
    dataframe['Generations no drug'] = pd.to_numeric(dataframe[\
                                        'Generations no drug'], errors='coerce')
    dataframe['Generations drug'] = pd.to_numeric(dataframe[\
                                        'Generations drug'],errors='coerce')
    dataframe['MM fraction'] = pd.to_numeric(dataframe['MM fraction'],
                                                            errors='coerce')

    # Make a meshgrid for the plot
    X_values = dataframe['Generations no drug'].unique()
    Y_values = dataframe['Generations drug'].unique()
    X_values, Y_values = np.meshgrid(X_values, Y_values)
    Z_values = np.zeros((20, 20))

    # Fill the 2D array with the MM fraction values by looping over each row
    for index, row in dataframe.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_values[j, i] = row.iloc[2]

    return (X_values, Y_values, Z_values)

def avarage_MMr_MMd_nr(dataframe, time, therapy):
    """ Function that calculates the average MMd and MMr number

    Parameters:
    -----------
    dataframe: Dataframe
        The dataframe containing the MMd and MMr numbers over time
    time: Int
        The time over which the average MMd and MMr number should be calculated
    therapy: String
        The kind of therapy used
    """

    last_MMd_fractions = dataframe['xMMd'].tail(int(time))
    average_MMd_fraction = round(last_MMd_fractions.sum() / time, 2)
    last_MMr_fractions = dataframe['xMMr'].tail(int(time))
    average_MMr_fraction = round(last_MMr_fractions.sum() / time, 2)
    print(f'{therapy}: xMMd =',average_MMd_fraction,
                                        'and xMMr =', average_MMr_fraction)

""" Figure to determine the difference between traditional and adaptive therapy"""
def Figure_continuous_MTD_vs_AT_s_and_w_a_h(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell type
    fraction dynamics by traditional therapy (continuous MTD) and adaptive
    therapy.The holiday and administration periods are 5 generations. It also
    prints the fraction values in the new equilibrium during adaptive and
    traditional therapy.

    Parameters:
    -----------
    n_switches: Int
        The fraction of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the fraction of time steps drugs are administared and the
        breaks are for the different Figures.
    """
    # Set initial parameter values
    N = 100
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    cOC_IH = 1.1
    cOB_IH = 0.7

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 1.4, 2.2, 1.5],
        [0.95, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 1.4, 2.2, 1.5],
        [0.95, 0.0, -0.5, -0.5],
        [0.55, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 1.4, 2.2, 1.5],
        [0.95, 0.0, -0.5, -0.5],
        [1.28, 0.0, 0.2, 0.0],
        [1.9, 0.0, -1.1, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.65

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 1.35

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(10, n_switches, t_steps_drug[0],
                t_steps_drug[0], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(10, n_switches, t_steps_drug[1],
                t_steps_drug[1], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(10, n_switches, t_steps_drug[2],
                t_steps_drug[2], xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_IH_comb,
                WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(10, 100, xOC, xOB, xMMd, xMMr, N, cOC,
            cOB, cMMd, cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(10, 100, xOC, xOB, xMMd, xMMr, N, cOC,
                            cOB, cMMd, cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH,
                            matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(10, 100, xOC, xOB, xMMd, xMMr, N, cOC,
                            cOB, cMMd, cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH,
                            matrix_IH_comb, WMMd_inhibitor_comb)

    # Print the equilibrium MMd and MMr values caused by the adaptive therapy


    avarage_MMr_MMd_nr(df_total_switch_GF, 10, 'Adaptive thearpy MMd GF IH')
    avarage_MMr_MMd_nr(df_total_switch_WMMd, 10, 'Adaptive thearpy WMMd IH')
    avarage_MMr_MMd_nr(df_total_switch_comb, 10, 'Adaptive thearpy IH combination')
    avarage_MMr_MMd_nr(df_total_GF, 10, 'Traditional thearpy MMd GF IH')
    avarage_MMr_MMd_nr(df_total_WMMd, 10, 'Traditional thearpy WMMd IH')
    avarage_MMr_MMd_nr(df_total_comb, 10, 'Traditional thearpy IH combination')


    # Save the data
    save_dataframe(df_total_switch_GF,'df_cell_frac_IH_switch_GF_IH_s_&_w_a_h.csv',
                                        r'..\data\data_model_frac_IH_inf')
    save_dataframe(df_total_switch_WMMd,'df_cell_frac_IH_switch_WMMd_IH_s_&_w_a_h.csv',
                                        r'..\data\data_model_frac_IH_inf')
    save_dataframe(df_total_switch_comb,'df_cell_frac_IH_switch_comb_IH_s_&_w_a_h.csv',
                                        r'..\data\data_model_frac_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_frac_IH_continuous_GF_IH_s_&_w_a_h.csv',
                                         r'..\data\data_model_frac_IH_inf')
    save_dataframe(df_total_WMMd,'df_cell_frac_IH_continuous_WMMd_IH_s_&_w_a_h.csv',
                                         r'..\data\data_model_frac_IH_inf')
    save_dataframe(df_total_comb,'df_cell_frac_IH_continuous_comb_IH_s_&_w_a_h.csv',
                                        r'..\data\data_model_frac_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[0, 0].set_xlim(1, 102)
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel(r'Cell fraction ($x_{i}$)', fontsize=12)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ", fontsize=14)
    axs[0, 0].grid(True, linestyle='--')

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                legend=False, ax=axs[0, 1])
    axs[0, 1].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[0, 1].set_xlim(1, 102)
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH", fontsize=14)
    axs[0, 1].grid(True, linestyle='--')

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[0, 2].set_xlim(1, 102)
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy IH combination", fontsize=14)
    axs[0, 2].grid(True, linestyle='--')

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[1, 0].set_xlim(1, 102)
    axs[1, 0].set_xlabel('Generations', fontsize=12)
    axs[1, 0].set_ylabel(r'Cell fraction ($x_{i}$)', fontsize=12)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH", fontsize=14)
    axs[1, 0].grid(True, linestyle='--')

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[1, 1].set_xlim(1, 102)
    axs[1, 1].set_xlabel('Generations', fontsize=12)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH", fontsize=14)
    axs[1, 1].grid(True, linestyle='--')

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['xOC', 'xOB', 'xMMd', 'xMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].axvspan(xmin=10, xmax=102, color='lightgray', alpha=0.45)
    axs[1, 2].set_xlim(1, 102)
    axs[1, 2].set_xlabel('Generations', fontsize=12)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy IH combination", fontsize=14)
    axs[1, 2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC fraction', 'OB fraction', 'MMd fraction',
                                                    'MMr fraction', 'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')
    save_Figure(plt, 'line_plot_cell_frac_IH_AT_MTD_s_&_w_a_h',
                                 r'..\visualisation\results_model_frac_IH_inf')
    plt.show()


""" 3D plot showing the best IH holiday and administration periods"""
def Figure_3D_MM_frac_IH_add_and_holiday():
    """ Figure that makes three 3D plot that shows the average MM fraction for
    different holiday and administration periods of only MMd GF inhibitor, only
    WMMd inhibitor or both. It prints the IH administration periods and holidays
    that caused the lowest total MM fraction."""

    # Set initial parameter values
    N = 100
    cMMr = 1.3
    cMMd = 1.2
    cOB = 0.8
    cOC = 1
    xOC = 0.2
    xOB = 0.3
    xMMd = 0.2
    xMMr = 0.3

    cOC_IH = 1.1
    cOB_IH = 0.7

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 1.4, 2.2, 1.6],
        [0.95, 0.0, -0.5, -0.5],
        [2.2, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 1.4, 2.2, 1.6],
        [0.95, 0.0, -0.5, -0.5],
        [0.73, 0, 0.2, 0.0],
        [1.9, 0, -0.8, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 1.4, 2.2, 1.6],
        [0.95, 0.0, -0.5, -0.5],
        [1.1, 0, 0.2, 0.0],
        [1.9, 0, -1.1, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.7

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 1.21

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holiday_GF_IH = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            frac_tumour = minimal_tumour_frac_t_steps(t_steps_drug,
                    t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                    cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_GF_IH)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                         'MM fraction': float(frac_tumour)}])
            df_holiday_GF_IH = combine_dataframes(df_holiday_GF_IH, new_row_df)

    # Save the data
    save_dataframe(df_holiday_GF_IH, 'df_cell_frac_IH_best_MMd_GF_IH_holiday.csv',
                                         r'..\data\data_model_frac_IH_inf')

    # Determine the axis values
    X_GF_IH, Y_GF_IH, Z_GF_IH = x_y_z_axis_values_3d_plot(df_holiday_GF_IH,
                                                                        'GF IH')

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holiday_W_IH = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            frac_tumour = minimal_tumour_frac_t_steps(t_steps_drug,
                t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd, cMMr,
                cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug':\
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                         'MM fraction': float(frac_tumour)}])
            df_holiday_W_IH = combine_dataframes(df_holiday_W_IH, new_row_df)

    # Save the data
    save_dataframe(df_holiday_W_IH, 'df_cell_frac_IH_best_WMMd_IH_holiday.csv',
                                         r'..\data\data_model_frac_IH_inf')

    # Determine the axis values
    X_W_IH, Y_W_IH, Z_W_IH = x_y_z_axis_values_3d_plot(df_holiday_W_IH, "W IH")

    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM fraction']
    df_holiday_comb = pd.DataFrame(columns=column_names)

    # Loop over al the t_step values for drug dministration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            frac_tumour = minimal_tumour_frac_t_steps(t_steps_drug,
                        t_steps_no_drug, xOC, xOB, xMMd, xMMr, N, cOC, cOB, cMMd,
                        cMMr, cOC_IH, cOB_IH, matrix_no_GF_IH, matrix_IH_comb,
                        WMMd_inhibitor_comb)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                            'MM fraction': float(frac_tumour)}])
            df_holiday_comb = combine_dataframes(df_holiday_comb, new_row_df)

    # Save the data
    save_dataframe(df_holiday_comb, 'df_cell_frac_IH_best_comb_IH_holiday.csv',
                                         r'..\data\data_model_frac_IH_inf')

    # Determine the axis values
    X_comb, Y_comb, Z_comb = x_y_z_axis_values_3d_plot(df_holiday_comb,
                                                            'IH combination')

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), subplot_kw={'projection': \
                        '3d'}, gridspec_kw={'hspace': 0.25, 'wspace': 0.25})

    # Plot each subplot
    for i, ax in enumerate(axes.flat, start=1):
        if i == 1:
            surf = ax.plot_surface(X_W_IH, Y_W_IH, Z_W_IH, cmap='coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM fraction')
            ax.set_title(r'A) $W_{MMd}$ IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 32, azim = -140)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IH')
            ax.set_ylabel('Generations IH')
            ax.set_zlabel('MM fraction')
            ax.set_title('B)  MMd GF IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 38, azim = -133)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('MM fraction')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_xlabel('Generations no IHs')
            ax.set_ylabel('Generations IHs')
            ax.set_zlabel('MM fraction')
            ax.set_title('C)  IH combination', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 43, azim = -148)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM fraction')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_frac_IH_best_IH_h_a_periods',
                                r'..\visualisation\results_model_frac_IH_inf')
    plt.show()

if __name__ == "__main__":
    main()
