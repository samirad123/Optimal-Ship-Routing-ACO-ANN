import streamlit as st
import numpy as np
import random
from geopy.distance import great_circle
import plotly.graph_objects as go
import pandas as pd

# Coordinates for the ports
port_coordinates = {
    "Mumbai": (18.9438, 72.8387),
    "Mormugao": (15.4000, 73.8000),
    "Mangalore": (12.8700, 74.8800),
    "Cochin": (9.9667, 76.2667)
}

ports = list(port_coordinates.keys())
n_ports = len(ports)

# Calculate distances between ports
distances = np.zeros((n_ports, n_ports))
for i in range(n_ports):
    for j in range(i + 1, n_ports):
        port1, port2 = ports[i], ports[j]
        coord1, coord2 = port_coordinates[port1], port_coordinates[port2]
        distance = great_circle(coord1, coord2).nautical
        distances[i][j] = distances[j][i] = distance


# Define the Ant Colony Optimization class
class AntColonyOptimization:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, evaporation_rate):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.n_ports = len(distances)
        self.pheromones = np.ones((self.n_ports, self.n_ports))

    def run(self, progress_callback):
        best_path = None
        best_cost = float('inf')

        for iteration in range(self.n_iterations):
            paths = self.construct_paths()
            self.update_pheromones(paths)

            iteration_best_path = min(paths, key=lambda x: self.path_cost(x))
            iteration_best_cost = self.path_cost(iteration_best_path)

            if iteration_best_cost < best_cost:
                best_path = iteration_best_path
                best_cost = iteration_best_cost

            progress_callback(iteration, paths, best_path, best_cost, self.pheromones)

        return best_path, best_cost

    def construct_paths(self):
        return [self.construct_single_path() for _ in range(self.n_ants)]

    def construct_single_path(self):
        path = [0]
        while path[-1] != self.n_ports - 1:
            current = path[-1]
            next_port = self.choose_next_port(current, path)
            path.append(next_port)
        return path

    def choose_next_port(self, current, path):
        unvisited = set(range(self.n_ports)) - set(path)
        if not unvisited:
            return self.n_ports - 1
        probabilities = self.calculate_probabilities(current, unvisited)
        return random.choices(list(unvisited), weights=probabilities)[0]

    def calculate_probabilities(self, current, available):
        probabilities = []
        for port in available:
            pheromone = self.pheromones[current][port]
            distance = self.distances[current][port]
            probability = (pheromone ** self.alpha) * ((1 / distance) ** self.beta)
            probabilities.append(probability)
        return probabilities

    def update_pheromones(self, paths):
        self.pheromones *= (1 - self.evaporation_rate)
        for path in paths:
            cost = self.path_cost(path)
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1 / cost
                self.pheromones[path[i + 1]][path[i]] += 1 / cost

    def path_cost(self, path):
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))


# Define additional functions for ANN-integrated ship routing
def initialize_pheromone():
    pheromone = np.ones((n_ports, n_ports))
    return pheromone


def train_ANN(historical_data):
    from keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if historical_data.empty:
        st.error("No historical data available for training.")
        return None

    # Print column names to debug
    st.write(historical_data.columns)

    # Replace with actual columns if they differ
    target_columns = ['Optimal Route', 'Route Cost']  # Adjust as needed
    feature_columns = [col for col in historical_data.columns if col not in target_columns]

    # Ensure target columns exist
    if not all(col in historical_data.columns for col in target_columns):
        st.error("Target columns are missing in the historical data.")
        return None

    X = historical_data[feature_columns]
    y = historical_data[target_columns]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    import keras
    # Define the ANN model
    model = keras.Sequential()
    model.add(layers.Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=len(target_columns), activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Save the trained model
    model.save('trained_ANN_model.h5')

    return model


def construct_path(start_port, end_port, obstacles, pheromone, ANN):
    path = [start_port]
    current_port = start_port
    while current_port != end_port:
        next_port = choose_next_port_with_ann(current_port, obstacles, pheromone, ANN)
        path.append(next_port)
        current_port = next_port
    return path


def choose_next_port_with_ann(current_port, obstacles, pheromone, ANN):
    # Placeholder logic for selecting the next port using ANN predictions and pheromones
    available_ports = set(range(n_ports)) - set(obstacles)
    if len(available_ports) == 0:
        return n_ports - 1
    probabilities = np.array([pheromone[current_port][port] for port in available_ports])
    probabilities /= probabilities.sum()  # Normalize to get probabilities
    next_port = random.choices(list(available_ports), weights=probabilities)[0]
    return next_port


def calculate_path_cost(path):
    return sum(distances[path[i]][path[i + 1]] for i in range(len(path) - 1))


def update_pheromone(path, path_cost, pheromone):
    for i in range(len(path) - 1):
        pheromone[path[i]][path[i + 1]] *= (1 - 0.1)  # rho = 0.1 as evaporation rate
        pheromone[path[i]][path[i + 1]] += 1 / path_cost


def find_best_route(pheromone):
    best_route = np.argmax(pheromone, axis=1)
    return best_route


def create_sea_route(start, end):
    lat1, lon1 = start
    lat2, lon2 = end

    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    curve_factor = 0.2
    delta_lat = (lat2 - lat1) * curve_factor
    mid_lon -= abs(delta_lat)

    return [start, (mid_lat, mid_lon), end]


def plot_route(paths, port_coordinates, pheromones):
    fig = go.Figure()

    for i in range(len(ports)):
        for j in range(i + 1, len(ports)):
            if pheromones[i][j] > 1:
                start = port_coordinates[ports[i]]
                end = port_coordinates[ports[j]]
                route = create_sea_route(start, end)
                lats, lons = zip(*route)
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=lons,
                    lat=lats,
                    line=dict(width=pheromones[i][j], color="rgba(255, 0, 0, 0.6)")
                ))

    port_lats, port_lons = zip(*port_coordinates.values())
    fig.add_trace(go.Scattermapbox(
        mode="markers+text",
        lon=port_lons,
        lat=port_lats,
        marker={'size': 10, 'color': 'blue'},
        text=list(port_coordinates.keys()),
        textposition="top center"
    ))

    for path in paths:
        route = []
        for i in range(len(path) - 1):
            start = port_coordinates[ports[path[i]]]
            end = port_coordinates[ports[path[i + 1]]]
            route.extend(create_sea_route(start, end))
        lats, lons = zip(*route)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=lons,
            lat=lats,
            line=dict(width=2, color="rgba(0, 255, 0, 0.6)")
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lon=np.mean(port_lons), lat=np.mean(port_lats)),
            zoom=5),
        showlegend=False,
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        height=600
    )

    return fig


# Streamlit UI
st.title('Real-time Ant Colony Optimization with ANN for Ship Routing')
st.write("Finding optimal sea route from Mumbai to Cochin")

st.sidebar.header('Algorithm Parameters')
n_ants = st.sidebar.slider('Number of Ants', 5, 50, 20)
n_iterations = st.sidebar.slider('Number of Iterations', 10, 100, 50)
alpha = st.sidebar.slider('Alpha (Pheromone Importance)', 0.1, 5.0, 1.0, 0.1)
beta = st.sidebar.slider('Beta (Distance Importance)', 0.1, 5.0, 2.0, 0.1)
evaporation_rate = st.sidebar.slider('Evaporation Rate', 0.01, 0.5, 0.1, 0.01)

map_placeholder = st.empty()
info_placeholder = st.empty()

if st.button('Run Simulation'):
    # Placeholder for historical data loading
    try:
        historical_data = pd.read_csv("Historical_data.csv")
    except FileNotFoundError:
        st.error("Historical data file not found.")
        historical_data = pd.DataFrame()  # Empty DataFrame as a placeholder

    # Train ANN model on historical data
    ANN = train_ANN(historical_data)
    
    # Initialize and run Ant Colony Optimization
    aco = AntColonyOptimization(
        distances,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate
    )

    def progress_callback(iteration, paths, best_path, best_cost, pheromones):
        info_placeholder.write(f"Iteration {iteration}: Best Cost = {best_cost}")

    best_path, best_cost = aco.run(progress_callback)

    # Plot the route
    fig = plot_route([best_path], port_coordinates, aco.pheromones)
    map_placeholder.plotly_chart(fig)
