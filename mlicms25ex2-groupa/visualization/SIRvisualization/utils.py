# -*- coding: utf-8 -*-

import os
import pandas as pd
import plotly.graph_objects as go


def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_REMOVED=2  # Add recovered pedestrian as function argument
                        ):
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted for visualization.

    Arguments:
    ----------
    df : Pandas.Dataframe
        Simulation output data from the csv file.
    ID_SUSCEPTIBLE : int
        ID of susceptible group in the Vadere processor file.
    ID_INFECTED : int
        ID of infected group in the Vadere processor file.
    ID_REMOVED : int
        ID of removed group in the Vadere processor file.
    Returns:
    --------
    group_counts : int
        Total number of SIR model groups to be visualized.
    """
    pedestrian_ids = df['pedestrianId'].unique()
    sim_times = df['simTime'].unique()
    group_counts = pd.DataFrame(columns=['simTime', 'group-s', 'group-i', 'group-r'])
    group_counts['simTime'] = sim_times
    group_counts['group-s'] = 0
    group_counts['group-i'] = 0
    group_counts['group-r'] = 0

    for pid in pedestrian_ids:
        simtime_group = df[df['pedestrianId'] == pid][['simTime', 'groupId-PID5']].values
        current_state = ID_SUSCEPTIBLE
        group_counts.loc[group_counts['simTime'] >= 0, 'group-s'] += 1
        for (st, g) in simtime_group:
            if g != current_state:  # Add state check to top for modularity
                if g == ID_INFECTED and current_state == ID_SUSCEPTIBLE:
                    group_counts.loc[group_counts['simTime'] > st, 'group-s'] -= 1
                    group_counts.loc[group_counts['simTime'] > st, 'group-i'] += 1
                    break
        current_state = ID_INFECTED  # Assign current state as if infected
        for (st, g) in simtime_group:  # Go through similar logic as if susceptible group
            if g != current_state:  # Add state check to top for modularity
                if g == ID_REMOVED and current_state == ID_INFECTED:  # If current state if different and is recovered
                    group_counts.loc[group_counts['simTime'] > st, 'group-i'] -= 1  # Decrease infected by one
                    group_counts.loc[group_counts['simTime'] > st, 'group-r'] += 1  # Increase recovered by one
                    break
    return group_counts


def create_folder_data_scatter(folder):
    """
    Arguments:
    ----------
    folder : String
        File path for the output folder of Vadere.
    Returns:
    --------
    [scatter_s, scatter_i, scatter_r] : list
        A list of values to be visualized for each group type.
    group_counts : int
        Total number of SIR model groups to be visualized.
    """
    file_path = os.path.join(folder, "SIRinformation.csv")
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path, delimiter=" ")

    print(data)

    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_REMOVED = 2

    # Change group counts to total 3 by adding recovered group
    group_counts = file_df_to_count_df(data, ID_REMOVED=ID_REMOVED, ID_INFECTED=ID_INFECTED, ID_SUSCEPTIBLE=ID_SUSCEPTIBLE)
    # group_counts.plot()
    scatter_s = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-s'],
                           name='susceptible ' + os.path.basename(folder),
                           mode='lines')
    scatter_i = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-i'],
                           name='infected ' + os.path.basename(folder),
                           mode='lines')
    # Add a scatter for recovered pedestrians the same way it's done for susceptible and infected ones
    scatter_r = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-r'],
                           name='recovered ' + os.path.basename(folder),
                           mode='lines')
    return [scatter_s, scatter_i, scatter_r], group_counts  # Return recovered pedestrians with others
