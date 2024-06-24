"""
Author:       Eva Nieuwenhuis
Student ID:   13717405
University:   UvA
Group:        Biosystems Data Analysis Group

Description:  The code of the model that simulates the dynamics in the multiple
              myeloma (MM) microenvironment with four cell types: drug-sensitive
              MM cells (MMd), resistant MM cells (MMr), osteoblasts (OB) and
              osteoclasts (OC). The model has collective interactions and is made
              in the framework of evolutionary game theory. In this model, there
              is looked at the number dynamics of the four cell types during
              different IH administration methods. Furthermore, it contains a
              (weighted) optimisation to find the best administration manner and
              combination. The IHs do not only influence on the MMd but also the
              OB and OC.

Example interaction matrix:
M = np.array([
         Foc     Fob   Fmmd   Fmmr
    OC  [b1,1,  b2,1,  b3,1,  b4,1],
    OB  [b1,2,  b2,2,  b3,2,  b4,2],
    MMd [b1,3,  b2,3,  b3,3,  b4,3],
    MMr [b1,4,  b2,4,  b3,4,  b4,4]])
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

    # Make a figure showing the cell number dynamics by traditional therapy and
    # by adaptive therapy
    list_t_steps_drug = [3, 3, 3]
    Figure_continuous_MTD_vs_AT_realistic(90, list_t_steps_drug)

    # Make a 3D figure showthing the effect of different drug holiday and
    # administration periods
    Figure_3D_MM_numb_IH_add_and_holiday()

    # Make a figure of the number dynamics whereby there is a limit for the MMd
    # and MMr number. The last paramter is low or high-> low: the MMr limit values
    # are low (< 500), high: the MMr limit values are high (> 500)
    Figure_AT_MMd_MMr_limit(500, 250)

    """ The optimisation """
    # Optimise IH administration duration, holiday duration and strength for
    # MMd GF IH -> WMMd IH -> holiday
    minimise_MM_GF_W_h_IH()

    # Optimise IH administration duration, holiday duration and strength for
    # WMMd IH -> MMd GF IH ->  holiday
    minimise_MM_W_GF_h_IH()

    # Optimise IH administration duration, holiday duration and strengths for
    # MMd GF IH -> IH combination -> WMMd IH -> holiday
    minimise_MM_GF_comb_W_h_IH()

    # Optimise IH administration duration, holiday duration and strengths for
    # WMMd IH -> IH combination -> MMd GF IH -> holiday
    minimise_MM_W_comb_GF_h_IH()

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths and MMd GF IH = 0.4
    minimise_MM_GF_W_h_changing_W_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_GF_W_h_changing_W_IH.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths and MMd GF IH = 0.4
    minimise_MM_W_GF_h_changing_W_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_W_GF_h_changing_W_IH.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths and WMMd IH = 0.4
    minimise_MM_GF_W_h_changing_GF_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_GF_W_h_changing_GF_IH.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths and WMMd IH = 0.4
    minimise_MM_W_GF_h_changing_GF_IH([0.8, 1.2, 0.3, 0.3], [0.7, 1.3, 0.3, 0.3],
    [0.9, 0.08, 0.2, 0.1], [1.0, 0.08, 0.2, 0.1], 'df_MM_W_GF_h_changing_GF_IH.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_W_IH([0.88, 1.32, 0.33, 0.33], [0.77, 1.43, 0.33,
                       0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_GF_W_h_changing_W_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_W_IH([0.88, 1.32, 0.33, 0.33], [0.77,1.43, 0.33,
                      0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_W_GF_h_changing_W_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_GF_IH([0.88, 1.32, 0.33, 0.33], [0.77,1.43, 0.33,
                      0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                        'df_MM_GF_W_h_changing_GF_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_GF_IH([0.88, 1.32, 0.33, 0.33], [0.77, 1.43, 0.33,
                       0.33], [0.99, 0.088, 0.22, 0.11], [1.1, 0.088, 0.22, 0.11],
                                      'df_MM_W_GF_h_changing_GF_IH_h_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate are
    # decreased with 10%
    minimise_MM_GF_W_h_changing_W_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                      0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                        'df_MM_GF_W_h_changing_W_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different WMMd IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_W_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                      0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                        'df_MM_W_GF_h_changing_W_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for MMd GF IH -> WMMd IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_GF_W_h_changing_GF_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                    0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                    'df_MM_GF_W_h_changing_GF_IH_l_gr_dr.csv')

    # Optimise IH administration and holiday duration for WMMd IH -> MMd GF IH ->
    # holiday for different MMd GF IH strengths whereby the growth and decay rate
    # are decreased with 10%
    minimise_MM_W_GF_h_changing_GF_IH([0.72, 1.08, 0.27, 0.27], [0.63, 1.17, 0.27,
                       0.27], [0.81, 0.072, 0.18, 0.09], [0.9, 0.072, 0.18, 0.09],
                                    'df_MM_W_GF_h_changing_GF_IH_l_gr_dr.csv')

    """The weighted optimisation"""
    # Optimise IH administration and holiday duration and strength for MMd GF IH
    # -> WMMd IH -> holiday where the weight of the MMr relative to the MMd can
    # be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strength for WMMd IH ->
    # MMd GF IH -> holiday where the weight of the MMr relative to the MMd can be
    # specified
    relative_weight_MMr = 1.2
    minimise_MM_W_GF_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strengths for MMd GF IH
    # -> IH combination -> WMMd IH -> holiday where the weight of the MMr relative
    # to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_GF_comb_W_h_IH_w(relative_weight_MMr)

    # Optimise IH administration and holiday duration and strengths for WMMd IH
    # -> IH combination -> MMd GF IH -> holiday where the weight of the MMr
    # relative to the MMd can be specified
    relative_weight_MMr = 1.2
    minimise_MM_W_comb_GF_h_IH_w(relative_weight_MMr)


def dOC_dt(nOC, nOB, nMMd, nMMr, gr_OC, dr_OC, matrix):
    """
    Function that calculates the change in number of osteoclasts.

    Parameters:
    -----------
    nOC: Float
         of OC.
    nOB: Float
         of OB.
    nMMd: Float
         of the MMd.
    nMMr: Float
         of the MMr.
    gr_OC: Float
        Growth rate of the OC.
    dr_OC: Float
        Dacay rate of the OC.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOC: Float
        Change in the number of OC.

    Example:
    -----------
    >>> dOC_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    744654.2266544278
    """
    # Extract the needed matrix values
    b1_1 = matrix[0, 0]
    b2_1 = matrix[0, 1]
    b3_1 = matrix[0, 2]
    b4_1 = matrix[0, 3]

    # Calculate the Change on in the number of OC
    change_nOC = (gr_OC * nOC**b1_1 * nOB**b2_1 * nMMd**b3_1 * nMMr**b4_1) - \
                                                                (dr_OC * nOC)
    return change_nOC

def dOB_dt(nOC, nOB, nMMd, nMMr, gr_OB, dr_OB, matrix):
    """ Function that calculates the change in the number of osteoblast.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_OB: Float
        Growth rate of the OB.
    dr_OB: Float
        Dacay rate of the OB.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nOB: Float
        Change in the number of OB.

    Example:
    -----------
    >>> dOB_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    1320.9296319483412
    """
    # Extract the necessary matrix values
    b1_2 = matrix[1, 0]
    b2_2 = matrix[1, 1]
    b3_2 = matrix[1, 2]
    b4_2 = matrix[1, 3]

    # Calculate the change in number of OB
    change_nOB = (gr_OB * nOC**b1_2 * nOB**b2_2 * nMMd**b3_2 * nMMr**b4_2) - \
                                                                    (dr_OB * nOB)
    return change_nOB

def dMMd_dt(nOC, nOB, nMMd, nMMr, gr_MMd, dr_MMd, matrix, WMMd_inhibitor = 0):
    """
    Function that calculates the change in the number of a drug-senstive MM cells.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
         Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_MMd: Float
        Growth rate of the MMd.
    dr_MMd: Float
        Decay rate of the MMd.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    change_nMMd: Float
        Change in the number of MMd.

    Example:
    -----------
    >>> dMMd_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    4198.444487046028
    """
    # Extract the necessary matrix values
    b1_3 = matrix[2, 0]
    b2_3 = matrix[2, 1]
    b3_3 = matrix[2, 2]
    b4_3 = matrix[2, 3]

    # Calculate the change in the number of MMd
    change_nMMd = (gr_MMd * nOC**b1_3 * nOB**b2_3 * nMMd**b3_3 * nMMr**b4_3 - \
                                    nMMd * WMMd_inhibitor) - (dr_MMd * nMMd)

    return change_nMMd

def dMMr_dt(nOC, nOB, nMMd, nMMr, gr_MMr, dr_MMr, matrix):
    """Function that calculates the change in the number of the MMr.

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    gr_MMr: Float
        Growth rate of the MMr.
    dr_MMr: Float
        Decay rate of the MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.

    Returns:
    --------
    change_nMMr: Float
        Change in the number of MMr.

    Example:
    -----------
    >>> dMMr_dt(10, 20, 10, 5, 0.8, 0.4, np.array([
    ...    [0.7, 1, 2.5, 2.1],
    ...    [1, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0, -0.2, 1.2]]))
    436.383290554087
    """
    # Extract the necessary matrix values
    b1_4 = matrix[3, 0]
    b2_4 = matrix[3, 1]
    b3_4 = matrix[3, 2]
    b4_4 = matrix[3, 3]

    # Calculate the change in the number of MMr
    change_nMMr = (gr_MMr * nOC**b1_4 * nOB**b2_4 * nMMd**b3_4 * nMMr**b4_4) - \
                                                                (dr_MMr * nMMr)
    return change_nMMr


def model_dynamics(y, t, growth_rates, decay_rates, matrix, WMMd_inhibitor = 0):
    """Function that determines the number dynamics in a population over time.

    Parameters:
    -----------
    y: List
        List with the values of nOC, nOB, nMMd and nMMr.
    t: Numpy.ndarray
        Array with all the time points.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    [nOC_change, nOB_change, nMMd_change, nMMr_change]: List
        List containing the changes in nOC, nOB, nMMd and nMMr.

    Example:
    -----------
    >>> model_dynamics([10, 20, 10, 5], 1, [0.8, 0.9, 1.3, 0.5],
    ...    [0.4, 0.3, 0.3, 0.6], np.array([
    ...    [0.7, 1.0, 2.5, 2.1],
    ...    [1.0, 1.4, -0.3, 1.0],
    ...    [2.5, 0.2, 1.1, -0.2],
    ...    [2.1, 0.0, -0.2, 1.2]]))
    [744654.2266544278, 1489.0458359418838, 6825.972291449797, 270.98955659630434]
    """
    nOC, nOB, nMMd, nMMr = y

    # Determine the change values
    nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0], decay_rates[0],
                                                                        matrix)
    nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1], decay_rates[1],
                                                                        matrix)
    nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2], decay_rates[2],
                                                        matrix, WMMd_inhibitor)
    nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3], decay_rates[3],
                                                                        matrix)

    # Make floats of the arrays
    nOC_change = float(nOC_change)
    nOB_change = float(nOB_change)
    nMMd_change = float(nMMd_change)
    nMMr_change = float(nMMr_change)

    return [nOC_change, nOB_change, nMMd_change, nMMr_change]


def dynamics_MMd_MMr_limits(time_IH, time_end, upper_limit_MMd, upper_limit_MMr,
            nOC, nOB, nMMd, nMMr, growth_rates, decay_rates, matrix_no_drugs,
            matrix_drugs, WMMd_inhibitor = 0):
    """Function that determines the number dynamics. It ensures that the MMr
    number and MMd number do not become too high.

    Parameters:
    -----------
    time_IH: Int
        Number of generations before the therapy start
    time_end: Int
        The last generation for which the numbers have to be calculated
    upper_limit_MMd: Int
        The maximum number of MMd, when reached the IH administration starts
    upper_limit_MMr: Int
        The maximum number of MMr, when reached the IH administration stops
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix_no_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when no IHs are given.
    matrix_drugs: Numpy.ndarray
        4x4 matrix containing the interaction factors when IHs are given.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_numbers: DataFrame
        The dataframe with the number per cell type over the time
    average_a_duration: Float
        The average administration duration
    average_h_duration: Float
        The average holiday duration
    """
    # Create a dataframe and lists
    df_numbers = pd.DataFrame(columns = ['Generation', 'nOC', 'nOB', 'nMMd',
                                                        'nMMr', 'total nMM'])
    duration_holiday = []
    duration_administration = []

    # Set the start values
    times_holiday = 0
    times_administration = 0
    duration = 0
    x = int(1)
    t = np.linspace(0, time_IH, time_IH)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_drugs)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
            'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_numbers = combine_dataframes(df_numbers, df_2)

    # Calculate the number of generations the therapy is given
    time_step_t = time_end - time_IH

    # Loop over the generations
    for time_step in range(time_step_t):

        # Determine the start numbers
        nOC = df_numbers['nOC'].iloc[-1]
        nOB = df_numbers['nOB'].iloc[-1]
        nMMd = df_numbers['nMMd'].iloc[-1]
        nMMr = df_numbers['nMMr'].iloc[-1]

        # If x = 1 add IHs and if x = 0 add no IHs
        if x == 1:
            # Determine the change values
            nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                decay_rates[0], matrix_drugs)
            nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                decay_rates[1], matrix_drugs)
            nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                decay_rates[2], matrix_drugs, WMMd_inhibitor)
            nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                decay_rates[3], matrix_drugs)

        if x == 0:
            # Determine the change values
            nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                decay_rates[0], matrix_no_drugs)
            nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                decay_rates[1], matrix_no_drugs)
            nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                                decay_rates[2], matrix_no_drugs)
            nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                decay_rates[3], matrix_no_drugs)

        # Calculate the new cell numbers
        nOC = nOC + nOC_change
        nOB = nOB + nOB_change
        nMMd = nMMd + nMMd_change
        nMMr = nMMr + nMMr_change
        nMMt = nMMd + nMMr

        # If there are too many MMr stop drug administration
        if nMMr > upper_limit_MMr and x == int(1) and duration > 5:

            # Add the administration duration to the list withd durations
            duration_administration.append(duration)
            times_administration += 1
            duration = 0

            # Start the IH holliday
            x = int(0)

        # If there are too many MMd stop drug holiday
        if nMMd > upper_limit_MMd and x == int(0) and duration > 5:

            # Add the holiday duration to the list withd durations
            duration_holiday.append(duration)
            times_holiday += 1
            duration = 0

            # Start the IH administration
            x = int(1)

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'Generation': time_IH+time_step, 'nOC': nOC,
        'nOB': nOB, 'nMMd': nMMd, 'nMMr': nMMr, 'total nMM': nMMt}])
        df_numbers = combine_dataframes(df_numbers, new_row_df)

        # Add one to the duration
        duration += 1

    # Calculate average administration and holiday duration
    if times_administration > 0 and times_holiday > 0:
        average_a_duration = sum(duration_administration) / times_administration
        average_h_duration = sum(duration_holiday) / times_holiday
    else:
        average_a_duration = sum(duration_administration)
        average_h_duration = sum(duration_holiday)

    return df_numbers, average_a_duration, average_h_duration

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

        # Return the dataframe that is not empty
        combined_df = df_1 if not df_1.empty else df_2

    else:
        # Delete the NA columns
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

def save_optimised_results(results, file_path):
    """ Function that saves the results of the optimised function as csv file.

    Parameters:
    -----------
    results: OptimizeResult
        The results of the scipy.optimize funtion
    file_path: String
        The name of the csv file and the path where the results will be saved.
    """
    # Extract the results
    optimised_para = results.x
    optimal_value = results.fun
    number_iterations = results.nit
    number_evaluations = results.nfev

    # Save the results in dictionary form
    results_to_saved = [ {"Optimised parameters": optimised_para.tolist(),
            "Optimal MM nr": optimal_value, 'nr iterations': number_iterations,
            'nr evaluations': number_evaluations}]

    with open(file_path, 'w', newline='') as csvfile:

        # Create header names
        header_names = ['Optimised parameters', 'Optimal MM nr', 'nr iterations',
                                                                'nr evaluations']
        writer = csv.DictWriter(csvfile, fieldnames = header_names)

        # Loop over the results
        writer.writeheader()
        for result in results_to_saved:
            writer.writerow(result)

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

def optimise_matrix():
    """Function that returns theinteraction matrices that are used for the
    optimisation. The matrices are for situations with and without IHs
    administered.

    Returns:
    -----------
    [matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb]: Numpy.ndarray
        The interaction matrices used during the optimisation.
    """

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    return matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb

def make_part_df(dataframe, start_time, time, growth_rates, decay_rates, matrix,
                WMMd_inhibitor = 0):
    """ Function that adds the cell numbers over a specified time to a given
    dataframe

    Parameters:
    -----------
    dataframe: DataFrame
        The dataframe to which the extra data should be added.
    start_time: Int
        The last generation in the current dataframe
    time: Int
        The time the cell number should be calculated
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total: DataFrame
        Dataframe with the extra nOC, nOB, nMMd and nMMr values
    """

    # Determine the start numbers
    nOC = dataframe['nOC'].iloc[-1]
    nOB = dataframe['nOB'].iloc[-1]
    nMMd = dataframe['nMMd'].iloc[-1]
    nMMr = dataframe['nMMr'].iloc[-1]

    t = np.linspace(start_time, start_time+ time, int(time))
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
        'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Add dataframe to total dataframe
    df_total= combine_dataframes(dataframe, df)
    df_total.reset_index(drop=True, inplace=True)

    return df_total

def start_df(t_steps, nOC, nOB, nMMd, nMMr, growth_rates, decay_rates,
             matrix_no_GF_IH):
    """ Function that maked a dataframe with the cell numbers over time

    Parameters:
    -----------
    t_steps:
        The number of generations the therapy is given
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """

    # Make a dataframe and set start parameter values
    df_total_switch = pd.DataFrame()
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0],
                            'nOB': y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3],
                            'total nMM': y[:, 3]+ y[:, 2]})
    return(df_total_switch)

def switch_dataframe(time_IH, n_switches, t_steps_drug, t_steps_no_drug, nOC,
            nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
            decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given time of drug holiday and administration periods.

    Parameters:
    -----------
    time_IH: Int
        The time point at witch the drugs are administered
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0

    # Make dataframe
    df_total_switch = start_df(time_IH, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates, matrix_no_GF_IH)

    # Increase the time
    time += time_IH

    # Perform a number of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_drug,
                growth_rates_IH, decay_rates_IH, matrix_GF_IH, WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix_no_GF_IH)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_GF_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
 t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
 decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, then a WMMd IH and then there
    is a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0

    # Make dataframe
    df_total_switch = start_df(30, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates, matrix_no_GF_IH)

    # Increase the time
    time += 30

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                                growth_rates_IH, decay_rates_IH, matrix_GF_IH)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # WMMd IH
        if x == 1:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Change the x and time value
            x = int(2)
            time += t_steps_WMMd_IH

        # No IH
        if x == 2:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix_no_GF_IH)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_W_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time. First a WMMd IH is administered, then a MMd GF IH and then there is a
    IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0

    # Make dataframe
    df_total_switch = start_df(30, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates, matrix_no_GF_IH)

    # Increase the time
    time += 30

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # MMd GF IH
        if x == 1:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                                growth_rates_IH, decay_rates_IH, matrix_GF_IH)

            # Change the x and time value
            x = int(2)
            time += t_steps_GF_IH

        # No IH
        if x == 2:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time,
                t_steps_no_drug, growth_rates, decay_rates, matrix_no_GF_IH)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_W_comb_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a IH combination, then a
    MMd GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0

    # Make dataframe
    df_total_switch = start_df(30, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates, matrix_no_GF_IH)

    # Increase the time
    time += 30

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix_no_GF_IH, WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # IH combination
        if x == 1:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_comb,
                                growth_rates_IH, decay_rates_IH, matrix_IH_comb,
                                WMMd_inhibitor_comb)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # MMd GF IH
        if x == 2:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                                growth_rates_IH, decay_rates_IH, matrix_GF_IH)

            # Change the x and time value
            x = int(3)
            time += t_steps_GF_IH

        # No IH
        if x == 3:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time,
                    t_steps_no_drug, growth_rates, decay_rates, matrix_no_GF_IH)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_GF_comb_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a MMd GF IH is administered, the a IH combination, then a
    MMd GF IH and then a IH holiday.

    Parameters:
    -----------
    n_rounds: Int
        The number of rounds of giving drugs and not giving drugs.
    t_steps_GF_IH: Int
        The number of generations MMD GF IH drugs are administared.
    t_steps_WMMd_IH: Int
        The number of generations WMMd IH drugs are administared.
    t_steps_comb: Int
        The number of generations WMMd IH and MMd GF IH drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd
        IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.
    WMMd_inhibitor_comb: Float
        The effect of a drug on the MMd fitness when also a MMd GF IH is given.

    Returns:
    --------
    df_total_switch: DataFrame
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0

    # Make dataframe
    df_total_switch = start_df(30, nOC, nOB, nMMd, nMMr, growth_rates,
                    decay_rates, matrix_no_GF_IH)

    # Increase the time
    time += 30

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                                growth_rates_IH, decay_rates_IH, matrix_GF_IH)

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # IH combination
        if x == 1:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_comb,
                                growth_rates_IH, decay_rates_IH, matrix_IH_comb,
                                WMMd_inhibitor_comb)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # WMMd IH
        if x == 2:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                            growth_rates_IH, decay_rates_IH, matrix_no_GF_IH,
                            WMMd_inhibitor)

            # Change the x and time value
            x = int(3)
            time += t_steps_WMMd_IH

        # No IH
        if x == 3:
            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time,
                    t_steps_no_drug, growth_rates, decay_rates, matrix_no_GF_IH)

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def minimal_tumour_nr_t_3_situations(t_steps_IH_strength, function_order, nOC,
                nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
                decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration and
    holiday duration and IH strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: Float
        The average total MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_no_drug, = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
     t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
     decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_round))
    average_MM_number = last_MM_numbers.sum() / (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_3_situations_IH(t_steps_IH_strength, function_order,
                weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
                decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given MMd GF IH administration, WMMd IH administration and holiday
    duration and IH strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no
        drugs are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.

    Returns:
    --------
    average_MM_number: float
        The average (weighted) MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_no_drug, GF_IH,\
                                            WMMd_inhibitor = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH
    matrix_GF_IH[2, 0] = 0.6 - GF_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
     t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
     decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)

def minimal_tumour_nr_t_4_situations_IH(t_steps_IH_strength, function_order,
    weight_MMr, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
    decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time for a given MMd GF IH administration, WMMd IH administration, IH
    combination administration and holiday duration and Ih strength.

    Parameters:
    -----------
    t_steps_IH_strength: List
        List with the number of generations the MMD GF IH, the WMMd IH and no drugs
        are administared and the MMD GF IH and WMMd IH strength.
    function_order: Function
        Function that makes a dataframe of the number values for a specific IH
        administration order.
    weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    matrix_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd

    Returns:
    --------
    average_MM_number: Float
        The average (weighted) MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug, GF_IH, \
         GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb = t_steps_IH_strength
    n_rounds = 50

    # Determine the round duration and the matrix values
    matrix_GF_IH[2, 0] = 0.6 - GF_IH
    matrix_IH_comb[2, 0] = 0.6 - GF_IH_comb
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH + t_steps_comb

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb,
        t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
        decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
        matrix_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == False:

        # Determine the average MM number in the last period
        last_MM_numbers = df['total nMM'].tail(int(time_round))
        average_MM_number = last_MM_numbers.sum() / (int(time_round))

    else:
        # Determine the weighted MM number in the last period
        last_MMd_numbers = df['nMMd'].tail(int(time_round))
        last_MMr_numbers = df['nMMr'].tail(int(time_round)) * weight_MMr
        average_MM_number = (last_MMd_numbers.sum() + last_MMr_numbers.sum())/ \
                                                                (int(time_round))

    return float(average_MM_number)

def dataframe_3D_plot(nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
 decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that create a dataframe with the average MM number for
    different IH administration and holiday durations

    Parameters:
    -----------
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_MM_frac: DataFrame
        The dataframe with the average MM number for different IH holiday
        and administration durations
    """
    # Make a dataframe
    column_names = ['Generations no drug', 'Generations drug', 'MM number']
    df_MM_nr = pd.DataFrame(columns=column_names)

    # Loop over all the t_step values for drug administration and drug holidays
    for t_steps_no_drug in range(2, 22):

        for t_steps_drug in range(2, 22):
            numb_tumour = minimal_tumour_numb_t_steps(t_steps_drug,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                    matrix_GF_IH, WMMd_inhibitor)

            # Add results to the dataframe
            new_row_df = pd.DataFrame([{'Generations no drug': \
                    int(t_steps_no_drug), 'Generations drug': int(t_steps_drug),
                                             'MM number': float(numb_tumour)}])
            df_MM_nr = combine_dataframes(df_MM_nr, new_row_df)

    return(df_MM_nr)

def continuous_add_IH_df(time_IH, end_generation, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the cell type numbers when the IHs
    are administered continuously.

    Parameters:
    -----------
    time_IH: Int
        The time point at which the IHs get administered
    end_generation: Int
        The last generation for which the numbers have to be calculated
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
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
        The dataframe with the cell numbers when IHs are continiously administered.
    """
    # Set the start values
    t = np.linspace(0, time_IH, time_IH)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_1 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Determine the current numbers
    nOC = df_1['nOC'].iloc[-1]
    nOB = df_1['nOB'].iloc[-1]
    nMMd = df_1['nMMd'].iloc[-1]
    nMMr = df_1['nMMr'].iloc[-1]

    # Set the currect values
    t = np.linspace(time_IH, end_generation, 400)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates_IH, decay_rates_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = combine_dataframes(df_1, df_2)

    return df_total

def x_y_z_axis_values_3d_plot(dataframe, name):
    """ Function that determines the x, y and z axis values from the given
    dataframe. It also prints the administration and holiday duration leading
    to the lowest total MM number in the equilibrium

    Parameters:
    -----------
    Dataframe: DataFrame
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
    # number
    min_index =  dataframe['MM number'].idxmin()
    g_no_drug_min = dataframe.loc[min_index, 'Generations no drug']
    g_drug_min = dataframe.loc[min_index, 'Generations drug']
    frac_min = dataframe.loc[min_index, 'MM number']

    print(f"""Lowest MM number: {frac_min}-> MMd {name} holidays are
            {g_no_drug_min} generations and MMd {name} administrations
            are {g_drug_min} generations""")

    # Avoid errors because of the wrong datatype
    dataframe['Generations no drug'] = pd.to_numeric(dataframe[\
                                        'Generations no drug'], errors='coerce')
    dataframe['Generations drug'] = pd.to_numeric(dataframe[\
                                        'Generations drug'],errors='coerce')
    dataframe['MM number'] = pd.to_numeric(dataframe['MM number'],
                                                            errors='coerce')

    # Make a meshgrid for the plot
    X_values = dataframe['Generations no drug'].unique()
    Y_values = dataframe['Generations drug'].unique()
    X_values, Y_values = np.meshgrid(X_values, Y_values)
    Z_values = np.zeros((20, 20))

    # Fill the 2D array with the MM number values by looping over each row
    for index, row in dataframe.iterrows():
        i = int(row.iloc[0]) - 2
        j = int(row.iloc[1]) - 2
        Z_values[j, i] = row.iloc[2]

    return (X_values, Y_values, Z_values)

def minimal_tumour_numb_t_steps(t_steps_drug, t_steps_no_drug, nOC, nOB, nMMd,
                nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values over
    time for a given time of a drug holiday.

    Parameters:
    -----------
    t_steps_drug: Int
        The number of generations drugs are administared.
    t_steps_no_drug: Int
        The number of generations drugs are not administared.
    nOC: Float
        Number of OC.
    nOB: Float
        Number of OB.
    nMMd: Float
        Number of the MMd.
    nMMr: Float
        Number of the MMr.
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    matrix_no_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when no GF IH are
        administered.
    matrix_GF_IH: Numpy.ndarray
        4x4 matrix containing the interaction factors when GF IH are administered.
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    average_MM_number: Float
        The average total MM number in the last period.

    """
    # Deteremine the number of switches
    time_step = (t_steps_drug + t_steps_no_drug) / 2
    n_switches = int((400 // time_step) -1)

    # Create a dataframe of the numbers
    df = switch_dataframe(30, n_switches, t_steps_drug, t_steps_no_drug, nOC,
                  nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates,
                  decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine the average MM number in the last period with and without drugs
    last_MM_numbers = df['total nMM'].tail(int(time_step *2))
    average_MM_number = last_MM_numbers.sum() / (int(time_step*2))

    return float(average_MM_number)

def avarage_MMr_MMd_nr(dataframe, time, therapy):
    """ Function that calculates the average MMd and MMr number

    Parameters:
    -----------
    dataframe: DataFrame
        The dataframe containing the MMd and MMr numbers over time
    time: Int
        The time over which the average MMd and MMr number should be calculated
    therapy: String
        The kind of therapy used
    """
    # Calculates the average MMd and MMr number
    last_MMd_numbers = dataframe['nMMd'].tail(int(time))
    average_MMd_number = round(last_MMd_numbers.sum() / time, 2)
    last_MMr_numbers = dataframe['nMMr'].tail(int(time))
    average_MMr_number = round(last_MMr_numbers.sum() / time, 2)
    print(f'{therapy}: nMMd =',average_MMd_number,
                                        'and nMMr =', average_MMr_number)

""" Figure to determine the difference between traditional and adaptive therapy
The interaction matrix is changed to make it more realistic"""
def Figure_continuous_MTD_vs_AT_realistic(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy. It
    also prints the number values in the new equilibrium during adaptive therapy.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 35
    nOB = 40
    nMMd = 35
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.6, 0.65]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.09, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.6, 0.65]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.6, 0.54],
        [0.3, 0.0, -0.3, -0.3],
        [0.22, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.8, 0.65]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.43

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 4.2

    # Make dataframe for the different drug hollyday duration values
    df_total_switch_GF = switch_dataframe(30, n_switches, t_steps_drug[0],
            t_steps_drug[0], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_switch_WMMd = switch_dataframe(30, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(30, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_IH_comb,
            WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                            growth_rates, growth_rates_IH, decay_rates,
                            decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH)
    df_total_WMMd = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_IH_comb, WMMd_inhibitor_comb)

    # Print the equilibrium MMd and MMr values caused by the adaptive therapy
    avarage_MMr_MMd_nr(df_total_switch_GF, 6, "Adaptive therapy MMd GF IH")
    avarage_MMr_MMd_nr(df_total_switch_WMMd, 6, "Adaptive therapy WMMd IH")
    avarage_MMr_MMd_nr(df_total_switch_comb, 6, "Adaptive therapy IH combination")

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH_r.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH_r.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 0].set_xlim(1, 302)
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=13)
    axs[0, 0].set_title(f"Traditional therapy MMd GF IH ", fontsize=14)
    axs[0, 0].set_yticks([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    axs[0, 0].grid(True, linestyle='--')

    # Plot the data without drug holidays in the second plot
    df_total_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 1])
    axs[0, 1].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 1].set_xlim(1, 302)
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel(' ')
    axs[0, 1].set_title(r"Traditional therapy $W_{MMd}$ IH", fontsize=14)
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 2].set_xlim(1, 302)
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy IH combination", fontsize=14)
    axs[0, 2].set_yticks([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    axs[0, 2].grid(True, linestyle='--')

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 0].set_xlim(1, 302)
    axs[1, 0].set_xlabel('Generations', fontsize=13)
    axs[1, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=13)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH", fontsize=14)
    axs[1, 0].grid(True, linestyle='--')
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 1].set_xlim(1, 302)
    axs[1, 1].set_xlabel('Generations', fontsize=13)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH", fontsize=14)
    axs[1, 1].grid(True, linestyle='--')

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 2].set_xlim(1, 302)
    axs[1, 2].set_xlabel('Generations', fontsize=13)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy IH combination", fontsize=14)
    axs[1, 2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC number', 'OB number', 'MMd number', 'MMr number',
                                                                    'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD_r',
                                 r'..\visualisation\results_model_nr_IH_inf')
    plt.show()

""" Figure to determine the difference between traditional and adaptive therapy
The AT administration and holiday durations depend on the MMd and MMr number"""
def Figure_AT_MMd_MMr_limit(upper_limit_MMd, upper_limit_MMr):
    """ Function that makes a figure with 3 subplots showing the cell number
    dynamics adaptive therapy. The IH administration starts when MMd because
    too high and stops when the MMr becomes too high. It prints the average
    adinistration and holiday duration

    Parameters:
    -----------
    upper_limit_MMd: Int
        The maximum number of MMd, when reached the IH administration starts
    upper_limit_MMr: Int
        The maximum number of MMr, when reached the IH administration stops
    """
    # Set start values
    nOC = 90
    nOB = 160
    nMMd = 100
    nMMr = 10
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.55, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.5, 0.7]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.08, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.5, 0.7]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.51, 0.51],
        [0.3, 0.0, -0.3, -0.3],
        [0.25, 0.0, 0.5, 0.0],
        [0.54, 0.0, -0.7, 0.7]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.24

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.55

    # Make dataframe for the different drug hollyday duration values
    df_switch_GF, a_dur_GF, h_dur_GF = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd,  upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_GF_IH)
    df_switch_WMMd, a_dur_W, h_dur_W = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd, upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_no_GF_IH, WMMd_inhibitor)
    df_switch_comb, a_dur_comb, h_dur_comb = dynamics_MMd_MMr_limits(30, 500,
        upper_limit_MMd, upper_limit_MMr, nOC, nOB, nMMd, nMMr, growth_rates,
        decay_rates, matrix_no_GF_IH, matrix_IH_comb, WMMd_inhibitor_comb)

    # Print average holiday and administration duration
    print(f"""The average MMd GF IH administration duration is
    {round(a_dur_GF, 0)} generations and the average MMd GF IH holiday duration
    is {round(h_dur_GF, 0)} generations""")
    print(f"""The average WMMd IH administration duration is {round(a_dur_W, 0)}
    generations and the average WMMd IH holiday duration is {round(h_dur_W, 0)}
    generations""")
    print(f"""The average IH combination administration duration is
    {round(a_dur_comb, 0)} generations and the average IH combination holiday
    duration is {round(h_dur_comb, 0)} generations""")

    # Save the data
    save_dataframe(df_switch_GF, 'df_cell_nr_GF_IH_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_switch_WMMd, 'df_cell_nr_W_IH_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')
    save_dataframe(df_switch_comb, 'df_cell_nr_IH_comb_AT_MM_limit.csv',
                                            r'..\data\data_model_nr_IH_inf')

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the data of the AT based on the MMd and MMr number (MMd GF IH)
    df_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0])
    axs[0].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[0].set_xlim(1, 402)
    axs[0].set_xlabel('Generations', fontsize=13)
    axs[0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=13)
    axs[0].set_title(f"Adaptive therapy MMd GF IH ", fontsize=14)
    axs[0].grid(True, linestyle='--')

    # Plot the data of the AT based on the MMd and MMr number (WMMd IH)
    df_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1])
    axs[1].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[1].set_xlim(1, 402)
    axs[1].set_xlabel('Generations', fontsize=13)
    axs[1].set_ylabel(' ')
    axs[1].set_title(r"Adaptive therapy $W_{MMd}$ IH", fontsize=14)
    axs[1].grid(True, linestyle='--')

    # Plot the data of the AT based on the MMd and MMr number (IH combination)
    df_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[2])
    axs[2].axvspan(xmin = 30, xmax = 402, color = 'lightgray', alpha = 0.45)
    axs[2].set_xlim(1, 402)
    axs[2].set_xlabel('Generations', fontsize=13)
    axs[2].set_ylabel(' ')
    axs[2].set_title(r"Adaptive therapy IH combination", fontsize=14)
    axs[2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC number', 'OB number', 'MMd number', 'MMr number',
                                                                    'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')

    # Save the figure
    save_Figure(plt, 'line_plot_cell_nr_AT_limit_MMd_MMr',
                                 r'..\visualisation\results_model_nr_IH_inf')

    plt.show()

""" 3D plot showing the best IH holiday and administration periods"""
def Figure_3D_MM_numb_IH_add_and_holiday():
    """ Figure that makes three 3D plot that shows the average number of MM for
    different holiday and administration periods of only MMd GF inhibitor, only
    WMMd inhibitor or both. It prints the IH administration periods and holidays
    that caused the lowest total MM number."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrix when no drugs are present
    matrix_no_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.6, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when only GF inhibitor drugs are present
    matrix_GF_IH = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.24, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # WMMd inhibitor effect when both inhibitor drugs are present
    WMMd_inhibitor_comb = 0.3

    # WMMd inhibitor effect when only WMMd IH is present
    WMMd_inhibitor = 0.42

    # Create a dataframe
    df_holiday_GF_IH = dataframe_3D_plot(nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH)

    # Save the data
    save_dataframe(df_holiday_GF_IH, 'df_cell_nr_IH_inf_best_MMd_GF_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_GF_IH, Y_GF_IH, Z_GF_IH = x_y_z_axis_values_3d_plot(df_holiday_GF_IH,
                                                                        "GF IH")

    # Create a dataframe
    df_holiday_W_IH = dataframe_3D_plot(nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_no_GF_IH, WMMd_inhibitor)

    # Save the data
    save_dataframe(df_holiday_W_IH, 'df_cell_nr_IH_inf_best_WMMd_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_W_IH, Y_W_IH, Z_W_IH = x_y_z_axis_values_3d_plot(df_holiday_W_IH, 'W IH')

    # Create a dataframe
    df_holiday_comb = dataframe_3D_plot(nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_IH_comb, WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_holiday_comb, 'df_cell_nr_IH_inf_best_comb_IH_holiday.csv',
                                             r'..\data\data_model_nr_IH_inf')

    # Determine the axis values
    X_comb, Y_comb, Z_comb = x_y_z_axis_values_3d_plot(df_holiday_comb,
                                                                "IH combination")

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), subplot_kw={'projection': '3d'},
                                    gridspec_kw={'hspace': 0.25, 'wspace': 0.25})

    # Plot each subplot
    for i, ax in enumerate(axes.flat, start=1):
        if i == 1:
            surf = ax.plot_surface(X_W_IH, Y_W_IH, Z_W_IH, cmap='coolwarm')

            # Add labels
            ax.set_ylabel('Generations admin')
            ax.set_xlabel('Generations holiday')
            ax.set_zlabel('MM number')
            ax.set_title(r'A) $W_{MMd}$ IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 12, azim = -138)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM number')

        elif i == 2:
            surf = ax.plot_surface(X_GF_IH, Y_GF_IH, Z_GF_IH, cmap = 'coolwarm')

            # Add labels
            ax.set_ylabel('Generations admin')
            ax.set_xlabel('Generations holiday')
            ax.set_zlabel('MM number')
            ax.set_title('B)  MMd GF IH', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 22, azim = -155)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')

            color_bar.set_label('MM number')

        elif i == 3:
            surf = ax.plot_surface(X_comb, Y_comb, Z_comb, cmap = 'coolwarm')

            # Add labels
            ax.set_ylabel('Generations admin')
            ax.set_xlabel('Generations holiday')
            ax.set_zlabel('MM number')
            ax.set_title('C)  IH combination', pad=10)

            # Turn to the right angle
            ax.view_init(elev = 25, azim = -126)

            # Add a color bar
            color_bar = fig.colorbar(surf, ax=ax, shrink=0.4, location= 'right')
            color_bar.set_label('MM number')

        else:
            # Hide the emply subplot
            ax.axis('off')

    # Add a color bar
    save_Figure(fig, '3d_plot_MM_nr_IH_inf_best_IH_h_a_periods',
                                r'..\visualisation\results_model_nr_IH_inf')
    plt.show()

"""optimise IH administration duration, holiday duration and strength for
MMd GF IH -> WMMd IH -> holiday """
def minimise_MM_GF_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> WMMd IH -> holiday -> MMd GF IH
    etc. It also determines the best MMd GF IH and WMMd IH strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.461, 2.434, 2.730, 0.382, 0.496]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH -> holiday.')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_GF_W_h_IH.csv')

"""Optimise IH administration duration, holiday duration and strength for
WMMd IH -> MMd GF IH -> holiday """
def minimise_MM_W_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> MMd GF IH -> holiday -> WMMd IH etc.
    It also determines the best MMd GF IH and WMMd IH strength."""
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.105, 2.065, 2.007, 0.321, 0.466]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, False,  nOC, nOB, nMMd, nMMr,
            growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_W_GF_h_IH.csv')

"""Optimise IH administration duration and holiday duration for WMMd IH-> IH
combination -> MMd GF IH -> holiday"""
def minimise_MM_W_comb_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH strength."""


    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.493, 3.227, 2.509, 3.520, 0.409, 0.085, 0.365, 0.089]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
     args=(switch_dataframe_W_comb_GF_h, False,  nOC, nOB, nMMd, nMMr, growth_rates,
     growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH,
     matrix_IH_comb), bounds = [(0, None), (0, None), (0, None), (0, None),
     (0, 0.6), (0, 0.6), (0, None), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_comb_GF_h_IH.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH-> IH
combination -> WMMd IH -> holiday"""
def minimise_MM_GF_comb_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.612, 2.135, 2.357, 2.288, 0.267, 0.088, 0.377, 0.106]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, False,  nOC, nOB, nMMd, nMMr,
        growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, 0.6), (0, 0.6), (0, None),
        (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_GF_comb_W_h_IH.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH -> MMd GF
IH -> holiday by different WMMd IH strengths"""
def minimise_MM_W_GF_h_changing_W_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different WMMd IH values when the order is WMMd IH -> MMd GF
    IH -> holiday -> WMMd IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_W_GF_h_change_W: DataFrame
        Dataframe with the MM number for different WMMd IH strengths.
    """

    # Make a datframe
    column_names = ['W IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_W_GF_h_change_W = pd.DataFrame(columns = column_names)

    for i in range(20):

        # Calculate and print the strength of the MMd GF IH
        W_IH = 0.2 + (round(i)/100)
        print(f'The WMMd IH stength is {W_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.2, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.001, 3.011]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_W_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, W_IH), bounds = [(0, None), (0, None), (0, None)],
                method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'W IH strength': W_IH, 'MMd GF IH duration':\
            result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
            result.x[2], 'MM number':result.fun}])
        df_W_GF_h_change_W = combine_dataframes(df_W_GF_h_change_W, new_row_df)

    # Save the data
    save_dataframe(df_W_GF_h_change_W, filename, r'..\data\data_model_nr_IH_inf')

    return(df_W_GF_h_change_W)


"""Optimise IH administration duration and holiday duration for WMMd IH -> MMd GF
IH -> holiday by different MMd GF IH strengths"""
def minimise_MM_W_GF_h_changing_GF_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different MMd GF IH values when the order is WMMd IH -> MMd GF
    IH -> holiday -> WMMd IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_W_GF_h_change_GF: DataFrame
        Dataframe with the MM number for different MMd GF IH strengths.
    """

    # Make a datframe
    column_names = ['GF IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_W_GF_h_change_GF = pd.DataFrame(columns = column_names)

    for i in range(14):

        # Calculate and print the strength of the MMd GF IH
        GF_IH = round(0.08 + (i/100), 2)
        print(f'The MMd GF IH stength is {GF_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6 -GF_IH, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        WMMd_inhibitor = 0.4

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.001, 3.011]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_W_GF_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH,WMMd_inhibitor), bounds = [(0, None), (0, None),
                (0, None)], method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'GF IH strength': GF_IH, 'MMd GF IH duration':\
            result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
            result.x[2], 'MM number':result.fun}])
        df_W_GF_h_change_GF = combine_dataframes(df_W_GF_h_change_GF, new_row_df)

    # Save the data
    save_dataframe(df_W_GF_h_change_GF, filename, r'..\data\data_model_nr_IH_inf')

    return(df_W_GF_h_change_GF)

"""optimise IH administration duration and holiday duration for MMd GF IH -> WMMd
IH -> holiday by different WMMd IH strengths"""
def minimise_MM_GF_W_h_changing_W_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different WMMd IH values when the order is MMd GF IH -> WMMd IH
    -> holiday -> MMd GF IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_GF_W_h_change_W: DataFrame
        Dataframe with the MM number for different WMMd IH strengths.
    """

    # Make a datframe
    column_names = ['W IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                                'Holiday duration', 'MM number']
    df_GF_W_h_change_W = pd.DataFrame(columns = column_names)

    for i in range(20):

        # Calculate and print the strength of the WMMd IH
        W_IH = 0.2 + (round(i)/100)
        print(f'The WMMd IH stength is {W_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.2, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.000, 3.000]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_GF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, W_IH), bounds = [(0, None), (0, None), (0, None)],
                method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: MMd GF IH -> WMMd IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'W IH strength': W_IH, 'MMd GF IH duration':\
                 result.x[0], 'WMMd IH duration':result.x[1],
                 'Holiday duration': result.x[2], 'MM number':result.fun}])
        df_GF_W_h_change_W = combine_dataframes(df_GF_W_h_change_W, new_row_df)
    # Save the data
    save_dataframe(df_GF_W_h_change_W, filename, r'..\data\data_model_nr_IH_inf')

    return(df_GF_W_h_change_W)

"""optimise IH administration duration and holiday duration for MMd GF IH -> WMMd
IH -> holiday by different MMd GF IH strengths"""
def minimise_MM_GF_W_h_changing_GF_IH(growth_rates, growth_rates_IH, decay_rates,
                                                     decay_rates_IH, filename):
    """Function that determines the best IH administration durations and holiday
    durations for different MMd GF IH values when the order is MMd GF IH -> WMMd
    IH -> holiday -> MMd GF IH etc.

    Parameters:
    -----------
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    growth_rates_IH: List
        List with the growth rate values of the OC, OB, MMd and MMr when a IH
        is administered.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    decay_rates_IH: List
        List with the decay rate values of OC, OB, MMd and MMr when a IH is
        administered.
    filename: String
        The name of the file in which the generated data is saved.

    Returns:
    --------
    df_GF_W_h_change_GF: DataFrame
        Dataframe with the MM number for different MMd GF IH strengths.
    """

    # Make a datframe
    column_names = ['GF IH strength', 'MMd GF IH duration', 'WMMd IH duration',
                                            'Holiday duration', 'MM number']
    df_GF_W_h_change_GF = pd.DataFrame(columns = column_names)

    for i in range(14):

        # Calculate and print the strength of the MMd GF IH
        GF_IH = round(0.08 + (i)/100, 2)
        print(f'The MMd GF IH stength is {GF_IH}')

        # Set start values
        nOC = 20
        nOB = 30
        nMMd = 20
        nMMr = 5

        # Payoff matrix when no drugs are present
        matrix_no_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        # Payoff matrix when only GF inhibitor drugs are present
        matrix_GF_IH = np.array([
            [0.0, 0.4, 0.65, 0.55],
            [0.3, 0.0, -0.3, -0.3],
            [0.6 - GF_IH, 0.0, 0.2, 0.0],
            [0.55, 0.0, -0.6, 0.4]])

        WMMd_inhibitor = 0.4

        # Optimise the administration and holiday durations
        # t_step_IH_strength = [GF IH t, W IH t, h t]
        t_step_IH_strength = [3.000, 2.000, 3.000]
        result = minimize(minimal_tumour_nr_t_3_situations, t_step_IH_strength,
                args=(switch_dataframe_GF_W_h, nOC, nOB, nMMd, nMMr, growth_rates,
                growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
                matrix_GF_IH, WMMd_inhibitor), bounds = [(0, None), (0, None),
                (0, None)], method='Nelder-Mead')

        # Print the results
        print('Optimising IH administration duration and holiday duration')
        print('Repeated order: MMd GF IH -> WMMd IH -> holiday')
        print(f"""The best MMd GF IH add duration is {result.x[0]} generations
        The best WMMd IH add duration is {result.x[1]} generations
        The best holiday duration is {result.x[2]} generations
        --> gives a MM number of {result.fun}""")

        # Add results to the dataframe
        new_row_df = pd.DataFrame([{'GF IH strength': GF_IH, 'MMd GF IH duration':\
                 result.x[0], 'WMMd IH duration':result.x[1], 'Holiday duration': \
                 result.x[2], 'MM number':result.fun}])
        df_GF_W_h_change_GF = combine_dataframes(df_GF_W_h_change_GF, new_row_df)

    # Save the data
    save_dataframe(df_GF_W_h_change_GF, filename, r'..\data\data_model_nr_IH_inf')

    return(df_GF_W_h_change_GF)


"""optimise IH administration duration, holiday duration and strength for
MMd GF IH -> WMMd IH -> holiday where the weight of the MMr relative to the MMd
can be specified """
def minimise_MM_GF_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH -> WMMd IH -> holiday -> MMd GF IH
    etc. It also determines the best MMd GF IH and WMMd IH strength. The weight
    of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.588, 2.778, 2.126, 0.351, 0.428]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> WMMd IH -> holiday.')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                        r'..\data\data_model_nr_IH_inf\optimise_GF_W_h_IH_w.csv')


"""Optimise IH administration duration, holiday duration and strength for
WMMd IH -> MMd GF IH -> holiday """
def minimise_MM_W_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> MMd GF IH -> holiday -> WMMd IH etc.
    It also determines the best MMd GF IH and WMMd IH strength. The weight of the
    MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """
    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.200, 2.339, 2.286, 0.476, 0.459]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
            nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
            matrix_no_GF_IH, matrix_GF_IH), bounds = [(0, None), (0, None),
            (0, None), (0, 0.6), (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best holiday duration is {result.x[2]} generations
    The best MMd GF IH strength is {result.x[3]}
    The best WMMd IH strengths is {result.x[4]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
                r'..\data\data_model_nr_IH_inf\optimise_W_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH-> IH
combination -> MMd GF IH -> holiday where the weight of the MMr relative to the
MMd can be specified"""
def minimise_MM_W_comb_GF_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH strength.
    The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    ## Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [3.765, 2.725, 2.090, 2.775, 0.411, 0.119, 0.439, 0.080]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_W_comb_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0.015, 0.6), (0, 0.6), (0, None),
        (0, None)], method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: WMMd IH -> IH combination -> MMd GF IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
        r'..\data\data_model_nr_IH_inf\optimise_W_comb_GF_h_IH_w.csv')

"""Optimise IH administration duration and holiday duration for MMd GF IH-> IH
combination -> WMMd IH -> holiday where the weight of the MMr relative to the
MMd can be specified"""
def minimise_MM_GF_comb_W_h_IH_w(relative_weight_MMr):
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength. The weight of the MMr relative to the MMd can be specified.

    Parameters:
    -----------
    relative_weight_MMr: Int
        The weight of the MMr relative to that of the MMd.
    """

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 5
    growth_rates = [0.8, 1.2, 0.3, 0.3]
    decay_rates = [0.9, 0.08, 0.2, 0.1]
    growth_rates_IH = [0.7, 1.3, 0.3, 0.3]
    decay_rates_IH = [1.0, 0.08, 0.2, 0.1]

    # Payoff matrices
    matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb = optimise_matrix()

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [3.293, 3.578, 2.209, 3.215, 0.443, 0.095, 0.365, 0.081]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, 0.6), (0, 0.6), (0, None), (0, None)],
        method='Nelder-Mead')

    # Print the results
    print('Optimising IH administration duration, holiday duration and strength')
    print('Repeated order: MMd GF IH -> IH combination -> WMMd IH -> holiday')
    print(f"""The best MMd GF IH add duration is {result.x[0]} generations
    The best WMMd IH add duration is {result.x[1]} generations
    The best IH combination duration is {result.x[2]} generations
    The best holiday duration is {result.x[3]} generations
    The best MMd GF IH strength when given alone is {result.x[4]}
    The best WMMd IH strength when given alone is {result.x[6]}
    The best MMd GF IH strength when given as a combination is {result.x[5]}
    The best WMMd IH strength when given as a combination is {result.x[7]}
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
        r'..\data\data_model_nr_IH_inf\optimise_GF_comb_W_h_IH_w.csv')

if __name__ == "__main__":
    main()
