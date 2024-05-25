"""
Author:            Eva Nieuwenhuis
University group:  Biosystems Data Analysis Group, UvA
Student ID:        13717405

Description:  The code of the model that simulates the dynamics in the multiple
              myeloma (MM) microenvironment with four cell types: drug-sensitive
              MM cells (MMd), resistant MM cells (MMr), osteoblasts (OB) and
              osteoclasts (OC). The model is a public goods game in the framework
              of evolutionary game theory with collective interactions. In this
              model, there is looked at the numbers of the four cell types.

              Compared to MM_model_nr_IH_inf.py there is a resistance mutation
              rate implemented. This mutation rate indicates how frequently MMd
              undergo mutations that make them resistant to the IHs, therefore
              transforming into MMr.


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
import doctest

def main():
    # Do doc tests
    doctest.testmod()
    #
    # # Make a figure showing the cell number dynamics by traditional therapy and
    # # by adaptive therapy
    # list_t_steps_drug = [3,3,3]
    # Figure_continuous_MTD_vs_AT_realistic(90, list_t_steps_drug)
    #
    #
    # """The optimisation"""
    # # Optimise IH administration duration, holiday duration and strength for
    # # MMd GF IH -> WMMd IH -> holiday
    # minimise_MM_GF_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strength for
    # # WMMd IH -> MMd GF IH ->  holiday
    # minimise_MM_W_GF_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # MMd GF IH -> IH combination -> WMMd IH -> holiday
    # minimise_MM_GF_comb_W_h_IH()
    #
    # # Optimise IH administration duration, holiday duration and strengths for
    # # WMMd IH -> IH combination -> MMd GF IH -> holiday
    # minimise_MM_W_comb_GF_h_IH()

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
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[0, 3]

    # Calculate the Change on in the number of OC
    change_nOC = (gr_OC * nOC**a * nOB**b * nMMd**c * nMMr**d) - (dr_OC * nOC)
    return change_nOC

def dOB_dt(nOC, nOB, nMMd, nMMr, gr_OB, dr_OB, matrix):
    """
    Function that calculates the change in the number of osteoblast.

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
    e = matrix[1, 0]
    f = matrix[1, 1]
    g = matrix[1, 2]
    h = matrix[1, 3]

    # Calculate the change in number of OB
    change_nOB = (gr_OB * nOC**e * nOB**f * nMMd**g * nMMr**h) - (dr_OB * nOB)
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
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]
    l = matrix[2, 3]

    # Calculate the change in the number of MMd
    change_nMMd = (gr_MMd * nOC**i * nOB**j * nMMd**k * nMMr**l - nMMd * \
                                            WMMd_inhibitor) - (dr_MMd * nMMd)

    return change_nMMd

def dMMr_dt(nOC, nOB, nMMd, nMMr, gr_MMr, dr_MMr, matrix):
    """
    Function that calculates the change in the number of the MMr.

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
    m = matrix[3, 0]
    n = matrix[3, 1]
    o = matrix[3, 2]
    p = matrix[3, 3]

    # Calculate the change in the number of MMr
    change_nMMr = (gr_MMr * nOC**m * nOB**n * nMMd**o * nMMr**p)-(dr_MMr * nMMr)
    return change_nMMr

def dOC_dt_no_MMr(nOC, nOB, nMMd, nMMr, gr_OC, dr_OC, matrix):
    """
    Function that calculates the change in the number of osteoclasts when no MMr
    are present.

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
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]

    # Calculate the Change on in the number of OC
    change_nOC = (gr_OC * nOC**a * nOB**b * nMMd**c) - (dr_OC * nOC)
    return change_nOC

def dOB_dt_no_MMr(nOC, nOB, nMMd, nMMr, gr_OB, dr_OB, matrix):
    """
    Function that calculates the change in the number of osteoblast when no MMr
    are present.

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
    e = matrix[1, 0]
    f = matrix[1, 1]
    g = matrix[1, 2]

    # Calculate the change in number of OB
    change_nOB = (gr_OB * nOC**e * nOB**f * nMMd**g) - (dr_OB * nOB)
    return change_nOB

def dMMd_dt_no_MMr(nOC, nOB, nMMd, nMMr, gr_MMd, dr_MMd, matrix, WMMd_inhibitor = 0):
    """
    Function that calculates the change in the number of a drug-senstive MM cells
    when no MMr are present.

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
    i = matrix[2, 0]
    j = matrix[2, 1]
    k = matrix[2, 2]

    # Calculate the change in the number of MMd
    change_nMMd = (gr_MMd * nOC**i * nOB**j * nMMd**k - nMMd * \
                                            WMMd_inhibitor) - (dr_MMd * nMMd)

    return change_nMMd

def mutation_MMd_to_MMr(IH_present, nMMd, nMMd_change, nMMr_change):
    """Function that determines the number of MMd that become a MMr through
    a mutation

    Parameters:
    -----------
    IH_present: Int
        Indicates if there is a IH present (0-> zero IHs present, 1 -> one IH
        present, 2 -> two IHs present)
    nMMd: Float
        Number of the MMd.
    nMMd_change: Float
        The change in the number of MMd.
    nMMr_change: Float
        The change in the number of MMr.

    Returns:
    --------
    nMMd_change: Float
        The change in the number of MMd after possible mutations.
    nMMr_change: Float
        The change in the number of MMr after possible mutations.


    Example:
    -----------
    >>> mutation_MMd_to_MMr(1, 100, 0.5, 0.4)
    (0.488, 0.41200000000000003)
    """
    # Determine the mutation rate based on how manny IHs are present
    if IH_present == 0:
        mutation_rate = 0.0001

    if IH_present == 1:
        mutation_rate = 0.00012

    if IH_present == 2:
        mutation_rate = 0.00007

    # Update the nMMd and nMMr change
    nMMd_change -= nMMd * mutation_rate
    nMMr_change += nMMd * mutation_rate

    return nMMd_change, nMMr_change

def model_dynamics(y, t, growth_rates, decay_rates, matrix, IH_present,
                                                            WMMd_inhibitor = 0):
    """Function that determines the number dynamics in a population over time.
    MMd can become MMr through mutations.

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
    IH_present: Int
        Indicates if there is a IH present (0-> zero IHs present, 1 -> one IH
        present, 2 -> two IHs present)
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
    ...    [2.1, 0.0, -0.2, 1.2]]), 1)
    [744654.2266544278, 1489.0458359418838, 6825.971091449797, 270.9907565963043]
    """
    nOC, nOB, nMMd, nMMr = y

    if nMMr == 0:
        # Determine the change values
        nOC_change = dOC_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                        decay_rates[0], matrix)
        nOB_change = dOB_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                        decay_rates[1], matrix)
        nMMd_change = dMMd_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                        decay_rates[2], matrix, WMMd_inhibitor)
        nMMr_change = 0

    else:
        # Determine the change values
        nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                        decay_rates[0], matrix)
        nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                        decay_rates[1], matrix)
        nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                        decay_rates[2], matrix, WMMd_inhibitor)
        nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                         decay_rates[3], matrix)


    # Determine the change in nMMd and nMMr based on the mutation rate
    nMMd_change, nMMr_change = mutation_MMd_to_MMr(IH_present, nMMd,
                                                      nMMd_change, nMMr_change)

    # Make floats of the arrays
    nOC_change = float(nOC_change)
    nOB_change = float(nOB_change)
    nMMd_change = float(nMMd_change)
    nMMr_change = float(nMMr_change)

    return [nOC_change, nOB_change, nMMd_change, nMMr_change]

def model_dynamics_no_mut(y, t, growth_rates, decay_rates, matrix, IH_present,
                                                            WMMd_inhibitor = 0):
    """Function that determines the number dynamics in a population over time.
    MMd cannot become MMr through mutations.

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
    IH_present: Int
        Indicates if there is a IH present (0-> zero IHs present, 1 -> one IH
        present, 2 -> two IHs present)
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
    ...    [2.1, 0.0, -0.2, 1.2]]), 1)
    [744654.2266544278, 1489.0458359418838, 6825.971091449797, 270.9907565963043]
    """
    nOC, nOB, nMMd, nMMr = y

    if nMMr == 0:
        # Determine the change values
        nOC_change = dOC_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                        decay_rates[0], matrix)
        nOB_change = dOB_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                        decay_rates[1], matrix)
        nMMd_change = dMMd_dt_no_MMr(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                        decay_rates[2], matrix, WMMd_inhibitor)
        nMMr_change = 0

    else:
        # Determine the change values
        nOC_change = dOC_dt(nOC, nOB, nMMd, nMMr, growth_rates[0],
                                                        decay_rates[0], matrix)
        nOB_change = dOB_dt(nOC, nOB, nMMd, nMMr, growth_rates[1],
                                                        decay_rates[1], matrix)
        nMMd_change = dMMd_dt(nOC, nOB, nMMd, nMMr, growth_rates[2],
                                        decay_rates[2], matrix, WMMd_inhibitor)
        nMMr_change = dMMr_dt(nOC, nOB, nMMd, nMMr, growth_rates[3],
                                                         decay_rates[3], matrix)

    # Make floats of the arrays
    nOC_change = float(nOC_change)
    nOB_change = float(nOB_change)
    nMMd_change = float(nMMd_change)
    nMMr_change = float(nMMr_change)

    return [nOC_change, nOB_change, nMMd_change, nMMr_change]

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


def make_part_df(dataframe, start_time, time, growth_rates, decay_rates, matrix,
                IH_present, WMMd_inhibitor = 0):
    """ Function that adds the cell numbers over a specified time to a given
    dataframe

    Parameters:
    -----------
    dataframe: DataFrame
        The dataframe to which the extra data should be added.
    start_time:
        The last generation in the current dataframe
    time: Int
        The time the cell number should be calculated
    growth_rates: List
        List with the growth rate values of the OC, OB, MMd and MMr.
    decay_rates: List
        List with the decay rate values of OC, OB, MMd and MMr.
    matrix: Numpy.ndarray
        4x4 matrix containing the interaction factors
    IH_present: Int
        The number of IHs present
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total: Dataframe
        Dataframe with the extra nOC, nOB, nMMd and nMMr values
    """

    # Determine the start numbers
    nOC = dataframe['nOC'].iloc[-1]
    nOB = dataframe['nOB'].iloc[-1]
    nMMd = dataframe['nMMd'].iloc[-1]
    nMMr = dataframe['nMMr'].iloc[-1]

    t = np.linspace(start_time, start_time+ time, int(time))
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix, IH_present, WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
        'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Add dataframe to total dataframe
    df_total= combine_dataframes(dataframe, df)
    df_total.reset_index(drop=True, inplace=True)

    return df_total

def switch_dataframe(time_IH, n_switches, t_steps_drug, t_steps_no_drug, nOC,
    nOB, nMMd, nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
    matrix_no_GF_IH, matrix_GF_IH, IH_present, WMMd_inhibitor = 0):
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
    IH_present: Int
        The number of IHs present
    WMMd_inhibitor: Float
        The effect of a drug on the MMd fitness.

    Returns:
    --------
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = time_IH
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0],
                            'nOB': y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3],
                            'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of switches
    for i in range(n_switches):

        # If x = 0 make sure the MMd is inhibited
        if x == 0:
            # Payoff matrix
            matrix = matrix_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_drug,
                            growth_rates_IH, decay_rates_IH, matrix, IH_present,
                            WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_drug

        # If x = 1 make sure the MMd is not inhibited
        else:
            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix, int(0))

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch


def switch_dataframe_GF_W_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                            t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                            growth_rates_IH, decay_rates, decay_rates_IH,
                            matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
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
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 60
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0],
                                'nOB': y[:, 1], 'nMMd': y[:, 2], 'nMMr': y[:, 3],
                                'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Payoff matrix
            matrix = matrix_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                            growth_rates_IH, decay_rates_IH, matrix, int(1))

            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # WMMd IH
        if x == 1:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix, int(1), WMMd_inhibitor)

            # Change the x and time value
            x = int(2)
            time += t_steps_WMMd_IH

        # No IH
        if x == 2:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix, int(0))

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_W_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
                    t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
                    growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor = 0):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a MMd GF IH and then there
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
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 60
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix, int(1), WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # MMd GF IH
        if x == 1:

            # Payoff matrix
            matrix = matrix_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                            growth_rates_IH, decay_rates_IH, matrix, int(1))

            # Change the x and time value
            x = int(2)
            time += t_steps_GF_IH

        # No IH
        if x == 2:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix, int(0))

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

def switch_dataframe_W_comb_GF_h(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
            t_steps_comb, t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates,
            growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
            matrix_GF_IH, matrix_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb):
    """ Function that makes a dataframe of the nOC, nOB, nMMd and nMMr values
    over time. First a WMMd IH is administered, then a IH combination, then a MMd
    GF IH and then a IH holiday.

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
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 60
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # WMMd IH
        if x == 0:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix, int(1), WMMd_inhibitor)

            # Change the x and time value
            x = int(1)
            time += t_steps_WMMd_IH

        # IH combination
        if x == 1:

            # Payoff matrix
            matrix = matrix_IH_comb

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_comb,
             growth_rates_IH, decay_rates_IH, matrix, int(2), WMMd_inhibitor_comb)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # MMd GF IH
        if x == 2:

            # Payoff matrix
            matrix = matrix_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                            growth_rates_IH, decay_rates_IH, matrix, int(1))

            # Change the x and time value
            x = int(3)
            time += t_steps_GF_IH

        # No IH
        if x == 3:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix, int(0))

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
    df_total_switch: Dataframe
        Dataframe with the nOC, nOB, nMMd and nMMr values over time.
    """
    # Set initial values
    x = 0
    time = 0
    df_total_switch = pd.DataFrame()
    t_steps = 60
    t = np.linspace(0, t_steps, t_steps*2)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_total_switch = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': \
        y[:, 1],'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Increase the time
    time += t_steps

    # Perform a number of rounds
    for i in range(n_rounds):

        # MMd GF IH
        if x == 0:

            # Payoff matrix
            matrix = matrix_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_GF_IH,
                            growth_rates_IH, decay_rates_IH, matrix, int(1))
                            
            # Change the x and time value
            x = int(1)
            time += t_steps_GF_IH

        # IH combination
        if x == 1:

            # Payoff matrix
            matrix = matrix_IH_comb

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_comb,
             growth_rates_IH, decay_rates_IH, matrix, int(2), WMMd_inhibitor_comb)

            # Change the x and time value
            x = int(2)
            time += t_steps_comb

        # WMMd IH
        if x == 2:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_WMMd_IH,
                growth_rates_IH, decay_rates_IH, matrix, int(1), WMMd_inhibitor)

            # Change the x and time value
            x = int(3)
            time += t_steps_WMMd_IH

        # No IH
        if x == 3:

            # Payoff matrix
            matrix = matrix_no_GF_IH

            # Extend the dataframe
            df_total_switch = make_part_df(df_total_switch, time, t_steps_no_drug,
                    growth_rates, decay_rates, matrix, int(0))

            # Change the x and time value
            x = int(0)
            time += t_steps_no_drug

    return df_total_switch

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
    n_rounds = 60

    # Determine the round duration and the matrix value
    time_round = t_steps_GF_IH + t_steps_no_drug + t_steps_WMMd_IH
    matrix_GF_IH[2, 0] = 0.6 - GF_IH

    # Create a dataframe of the numbers
    df = function_order(n_rounds, t_steps_GF_IH, t_steps_WMMd_IH,
      t_steps_no_drug, nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
      decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, WMMd_inhibitor)

    # Determine if the normal or weighted MMM number should be calculated
    if weight_MMr == int(1):

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
    matrix_IH_comb: Numpy.ndarray
        4x4 matrix containing the interaction factors when MMd GF IH and a WMMd

    Returns:
    --------
    average_MM_number: float
        The average (weighted) MM number in the last period.
    """
    # Unpack the values that should be optimised
    t_steps_GF_IH, t_steps_WMMd_IH, t_steps_comb, t_steps_no_drug, GF_IH, \
         GF_IH_comb, WMMd_inhibitor, WMMd_inhibitor_comb = t_steps_IH_strength
    n_rounds = 60

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
    if weight_MMr == int(1):

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

def continuous_add_IH_df(time_IH, end_generation, nOC, nOB, nMMd,nMMr,
                growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                matrix_no_GF_IH, matrix_GF_IH, IH_present, WMMd_inhibitor = 0):
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
    IH_present: Int
        Indicates if there is a IH present (0-> zero IHs present, 1 -> one IH
        present, 2 -> two IHs present)
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
    parameters = (growth_rates, decay_rates, matrix_no_GF_IH, int(0))

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
    t = np.linspace(time_IH, end_generation, 200)
    y0 = [nOC, nOB, nMMd, nMMr]
    parameters = (growth_rates_IH, decay_rates_IH, matrix_GF_IH, IH_present,
                                                    WMMd_inhibitor)

    # Determine the ODE solutions
    y = odeint(model_dynamics, y0, t, args=parameters)
    df_2 = pd.DataFrame({'Generation': t, 'nOC': y[:, 0], 'nOB': y[:, 1],
                'nMMd': y[:, 2], 'nMMr': y[:, 3], 'total nMM': y[:, 3]+ y[:, 2]})

    # Combine the dataframes
    df_total = combine_dataframes(df_1, df_2)

    return df_total

""" Figure to determine the difference between traditional and adaptive therapy
The interaction matrix is changed to make it more realistic"""
def Figure_continuous_MTD_vs_AT_realistic(n_switches, t_steps_drug):
    """ Function that makes a figure with 6 subplots showing the cell number
    dynamics by traditional therapy (continuous MTD) and adaptive therapy.

    Parameters:
    -----------
    n_switches: Int
        The number of switches between giving drugs and not giving drugs.
    t_steps_drug: List
        List with the number of time steps drugs are administared and the breaks
        are for the different Figures.
    """
    # Set start values
    nOC = 220
    nOB = 310
    nMMd = 210
    nMMr = 0
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
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, int(1))
    df_total_switch_WMMd = switch_dataframe(30, n_switches, t_steps_drug[1],
            t_steps_drug[1], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_no_GF_IH,
            int(1), WMMd_inhibitor)
    df_total_switch_comb = switch_dataframe(30, n_switches, t_steps_drug[2],
            t_steps_drug[2], nOC, nOB, nMMd, nMMr, growth_rates, growth_rates_IH,
            decay_rates, decay_rates_IH, matrix_no_GF_IH, matrix_IH_comb,
            int(2), WMMd_inhibitor_comb)

    # Make dataframes for continiously administration
    df_total_GF = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                        growth_rates, growth_rates_IH, decay_rates,
                        decay_rates_IH, matrix_no_GF_IH, matrix_GF_IH, int(1))
    df_total_WMMd = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                    growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                    matrix_no_GF_IH, matrix_no_GF_IH, int(1), WMMd_inhibitor)
    df_total_comb = continuous_add_IH_df(30, 300, nOC, nOB, nMMd, nMMr,
                growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
                matrix_no_GF_IH, matrix_IH_comb, int(2), WMMd_inhibitor_comb)

    # Save the data
    save_dataframe(df_total_switch_GF, 'df_cell_nr_IH_inf_switch_GF_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')
    save_dataframe(df_total_switch_WMMd, 'df_cell_nr_IH_inf_switch_WMMd_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')
    save_dataframe(df_total_switch_comb, 'df_cell_nr_IH_inf_switch_comb_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')
    save_dataframe(df_total_GF, 'df_cell_nr_IH_inf_continuous_GF_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')
    save_dataframe(df_total_WMMd, 'df_cell_nr_IH_inf_continuous_WMMd_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')
    save_dataframe(df_total_comb, 'df_cell_nr_IH_inf_continuous_comb_IH_r.csv',
                                    r'..\data\data_model_nr_IH_inf_mutation')

    # Create a Figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))

    # Plot the data without drug holidays in the first plot
    df_total_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 0])
    axs[0, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 0].set_xlim(1, 302)
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=12)
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
    axs[0, 1].set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    axs[0, 1].grid(True, linestyle='--')

    # Plot the data without drug holidays in the third plot
    df_total_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[0, 2])
    axs[0, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[0, 2].set_xlim(1, 302)
    axs[0, 2].set_xlabel(' ')
    axs[0, 2].set_ylabel(' ')
    axs[0, 2].set_title(r"Traditional therapy IH combination", fontsize=14)
    axs[0, 2].grid(True, linestyle='--')

    # Plot the data with drug holidays in the fourth plot
    df_total_switch_GF.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 0])
    axs[1, 0].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 0].set_xlim(1, 302)
    axs[1, 0].set_xlabel('Generations', fontsize=12)
    axs[1, 0].set_ylabel(r'Cell number ($n_{i}$)', fontsize=12)
    axs[1, 0].set_title(f"Adaptive therapy MMd GF IH", fontsize=14)
    axs[1, 0].grid(True, linestyle='--')
    plt.grid(True)

    # Plot the data with drug holidays in the fifth plot
    df_total_switch_WMMd.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 1])
    axs[1, 1].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 1].set_xlim(1, 302)
    axs[1, 1].set_xlabel('Generations', fontsize=12)
    axs[1, 1].set_ylabel(' ')
    axs[1, 1].set_title(r"Adaptive therapy $W_{MMd}$ IH", fontsize=14)
    axs[1, 1].grid(True, linestyle='--')

    # Plot the data with drug holidays in the sixth plot
    df_total_switch_comb.plot(x='Generation', y=['nOC', 'nOB', 'nMMd', 'nMMr'],
                    color= ['tab:pink', 'tab:purple', 'tab:blue', 'tab:red'],
                                                    legend=False, ax=axs[1, 2])
    axs[1, 2].axvspan(xmin = 30, xmax = 302, color = 'lightgray', alpha = 0.45)
    axs[1, 2].set_xlim(1, 302)
    axs[1, 2].set_xlabel('Generations', fontsize=12)
    axs[1, 2].set_ylabel(' ')
    axs[1, 2].set_title(r"Adaptive therapy IH combination", fontsize=14)
    axs[1, 2].grid(True, linestyle='--')

    # Create a single legend outside of all plots
    legend_labels = ['OC number', 'OB number', 'MMd number', 'MMr number',
                                                                    'Therapy']
    fig.legend(labels = legend_labels, loc='upper center', ncol=5,
                                                            fontsize='x-large')
    save_Figure(plt, 'line_plot_cell_nr_IH_inf_AT_MTD_r',
                            r'..\visualisation\results_model_nr_IH_inf_mutation')
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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.160, 3.525, 2.255, 0.357, 0.308]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_GF_W_h, int(1), nOC, nOB, nMMd, nMMr,
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
                r'..\data\data_model_nr_IH_inf_mutation\optimise_GF_W_h_IH.csv')


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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [3.052, 2.168, 2.045, 0.472, 0.348]
    result = minimize(minimal_tumour_nr_t_3_situations_IH, t_step_IH_strength,
            args=(switch_dataframe_W_GF_h, int(1), nOC, nOB, nMMd, nMMr,
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
                r'..\data\data_model_nr_IH_inf_mutation\optimise_W_GF_h_IH.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> MMd GF IH -> holiday"""
def minimise_MM_W_comb_GF_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is WMMd IH -> IH combination -> MMd GF IH -> holiday
    -> WMMd IH etc.It also determines the best MMd GF IH and WMMd IH strength."""


    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.056, 2.940, 3.020, 2.107, 0.469, 0.105, 0.321, 0.109]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
     args=(switch_dataframe_W_comb_GF_h, int(1), nOC, nOB, nMMd, nMMr,
     growth_rates, growth_rates_IH, decay_rates, decay_rates_IH, matrix_no_GF_IH,
     matrix_GF_IH, matrix_IH_comb), bounds = [(0, None), (0, None), (0, None),
     (0, None), (0, 0.6), (0, 0.6), (0, None), (0, None)], method='Nelder-Mead')

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
        r'..\data\data_model_nr_IH_inf_mutation\optimise_W_comb_GF_h_IH.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> WMMd IH -> holiday"""
def minimise_MM_GF_comb_W_h_IH():
    """Function that determines the best IH administration durations and holiday
    durations when the order is MMd GF IH-> IH combination -> WMMd IH -> holiday
    -> MMd GF IH etc.It also determines the best MMd GF IH and WMMd IH
    strength."""

    # Set start values
    nOC = 20
    nOB = 30
    nMMd = 20
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.202, 2.263, 2.344, 2.435, 0.368, 0.094, 0.356, 0.084]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, int(1), nOC, nOB, nMMd, nMMr,
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
            r'..\data\data_model_nr_IH_inf_mutation\optimise_GF_comb_W_h_IH.csv')

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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.138, 2.858, 2.438, 0.333, 0.389]
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
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
            r'..\data\data_model_nr_IH_inf_mutation\optimise_GF_W_h_IH_w.csv')


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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, h t, GF IH s, W IH s]
    t_step_IH_strength = [2.134, 2.503, 2.417, 0.417, 0.351]
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
                r'..\data\data_model_nr_IH_inf_mutation\optimise_W_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for WMMd IH->
IH combination -> MMd GF IH -> holiday where the weight of the MMr relative to the
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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [3.328, 2.421, 2.058, 2.319, 0.494, 0.103, 0.311, 0.096]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_W_comb_GF_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
        matrix_no_GF_IH, matrix_GF_IH, matrix_IH_comb), bounds = [(0, None),
        (0, None), (0, None), (0, None), (0, 0.6), (0, 0.6), (0, None),
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
        r'..\data\data_model_nr_IH_inf_mutation\optimise_W_comb_GF_h_IH_w.csv')


"""Optimise IH administration duration and holiday duration for MMd GF IH->
IH combination -> WMMd IH -> holiday where the weight of the MMr relative to the
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
    nMMr = 0
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
        [0.2, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.6, 0.4]])

    # Payoff matrix when both inhibitor drugs are present
    matrix_IH_comb = np.array([
        [0.0, 0.4, 0.65, 0.55],
        [0.3, 0.0, -0.3, -0.3],
        [0.4, 0.0, 0.2, 0.0],
        [0.55, 0.0, -0.8, 0.4]])

    # Optimise the administration and holiday durations and the IH strengths
    # t_step_IH_strength = [GF IH t, W IH t, comb t, h t, GF IH s, comb GF IH s
    # W IH s, comb W IH s]
    t_step_IH_strength = [2.422, 2.113, 2.003, 2.769, 0.405, 0.081, 0.314, 0.112]
    result = minimize(minimal_tumour_nr_t_4_situations_IH, t_step_IH_strength,
        args=(switch_dataframe_GF_comb_W_h, relative_weight_MMr, nOC, nOB, nMMd,
        nMMr, growth_rates, growth_rates_IH, decay_rates, decay_rates_IH,
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
    --> gives a weighted MM number of {result.fun}""")

    # Save the results
    save_optimised_results(result,
        r'..\data\data_model_nr_IH_inf_mutation\optimise_GF_comb_W_h_IH_w.csv')

if __name__ == "__main__":
    main()
