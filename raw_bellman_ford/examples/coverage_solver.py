"""Module containing drone coverage example."""

import pandas as pd
from raw_bellman_ford.algorithms.h_bellman_ford import HBellmanFord, HypergraphMetrics

def check_hypergraph_coverage(drone_data, diameter):
    """
    Check the coverage of a hypergraph by drones.
    
    Args:
        drone_data (list): A list of dictionaries containing drone data.
            Each dictionary represents a drone and contains the following keys:
            - 'Максимальный диаметр обзора (км)' (float): The maximum viewing diameter of the drone in kilometers.
        
        diameter (float): The minimum required coverage diameter in kilometers.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the drone data with an additional column 'Соответствует' (str).
            The 'Соответствует' column indicates whether each drone's maximum viewing diameter meets the coverage requirement.
            If a drone's maximum viewing diameter is greater than or equal to the required diameter, the value is 'Да'.
            Otherwise, the value is 'Нет'.
    """
    drone_df = pd.DataFrame(drone_data)
    
    drone_df['Соответствует'] = 'Нет'
    
    for index, drone in drone_df.iterrows():
        if drone['Максимальный диаметр обзора (км)'] >= diameter:
                drone_df.at[index, 'Соответствует'] = 'Да'
                break
    
    return drone_df

drone_data = {
    '№': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'Название БПЛА': [
        'DJI Phantom 4 RTK', 'DJI Inspire 2', 'Autel Robotics EVO Lite+', 'Parrot Anafi USA', 'Yuneec Typhoon H3',
        'Skydio 2', 'Freefly Alta X', 'SenseFly eBee X', 'PowerVision PowerEgg X', 'Autel Robotics Dragonfish'
    ],
    'Максимальный диаметр обзора (км)': [0.3, 1.2, 0.8, 0.6, 1.5, 0.7, 1.0, 1.8, 0.4, 1.3],
}

json_hypergraph = {
    "nodes": {
        1: {"type": 1, "weight": None},
        2: {"type": 1, "weight": None},
        3: {"type": 1, "weight": None},
        4: {"type": 1, "weight": None},
        5: {"type": 1, "weight": None},
        6: {"type": 1, "weight": None},
        7: {"type": 1, "weight": None},
        8: {"type": 1, "weight": None},
        9: {"type": 1, "weight": None},
        10: {"type": 1, "weight": None},
    },
    "edges": [
        {"nodes": (1, 2), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (2, 3), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (1, 4), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (3, 4), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (7, 8), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (8, 9), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (9, 10), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (10, 7), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((1, 2, 3, 4), 5), "weight": 0.2, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((1, 2, 3, 4), 6), "weight": 0.8, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (5, 6), "weight": 0.7, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((7, 8, 9, 10), 6), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((7, 8, 9, 10), (1, 2, 3, 4)), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
    ],
    "hyperedges": {
        (1, 2, 3, 4): {"type": 1},
        (7, 8, 9, 10): {"type": 1},
    },
}

hypergraph_from_json = HBellmanFord(**json_hypergraph)

metrics_calculator = HypergraphMetrics(hypergraph_from_json)

metrics_calculator.print_matrix()

metrics_calculator.print_eccentricities()

metrics_calculator.print_centralities()

metrics_calculator.print_central_and_peripheral_nodes()

diameter, radius = metrics_calculator.print_diameter_and_radius()

result_df = check_hypergraph_coverage(drone_data, diameter)

print(result_df)
