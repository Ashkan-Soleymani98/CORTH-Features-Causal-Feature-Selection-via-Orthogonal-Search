#!/usr/bin/python
import sys

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import os


def plot_graph(adjacency_matrix, path):
    nodes_num = len(adjacency_matrix)
    G = nx.MultiDiGraph()
    G.add_nodes_from([i for i in range(nodes_num)])
    for i in range(nodes_num):
        for j in range(nodes_num):
            if adjacency_matrix[i][j] != 0:
                G.add_weighted_edges_from([(i, j, adjacency_matrix[i][j])])

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    pos = nx.circular_layout(G)
    labels = {i: ("x" + str(i) if i != 0 else "y") for i in range(nodes_num)}
    # print(labels)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    nx.draw(G, pos=pos, node_color='y', edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.Reds_r,
            node_size=500)
    plt.savefig(str(path) + '/DAG' + '.png')
    plt.show()


def get_linear_parents(adjacency_matrix, node):
    nodes_num = len(adjacency_matrix)
    parents = list()
    for i in range(nodes_num):
        if adjacency_matrix[i][node] == 1:
            parents.append(i)
    return parents

def get_nonlinear_parents(adjacency_matrix, node):
    nodes_num = len(adjacency_matrix)
    parents = list()
    for i in range(nodes_num):
        if adjacency_matrix[i][node] == 2:
            parents.append(i)
    return parents


def set_values(adjacency_matrix, order, mu, sigma, theta, nonlinear_probability, b, a):
    nodes_num = len(adjacency_matrix)
    node_values = [np.random.normal(mu, sigma) for i in range(nodes_num)]
    for node in order:
        parents = get_linear_parents(adjacency_matrix, node)
        for par in parents:
            node_values[node] += node_values[par] * theta
        parents = get_nonlinear_parents(adjacency_matrix, node)
        for par in parents:
            node_values[node] += a * math.tanh(node_values[par] * b)
    return node_values


def plot_graph_values(adjacency_matrix, values, iteration_number, path):
    nodes_num = len(adjacency_matrix)
    G = nx.MultiDiGraph()
    G.add_nodes_from([i for i in range(nodes_num)])
    for i in range(nodes_num):
        for j in range(nodes_num):
            if adjacency_matrix[i][j] != 0:
                G.add_weighted_edges_from([(i, j, adjacency_matrix[i][j])])

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    pos = nx.circular_layout(G)
    labels = {i: str(round(values[i], 3)) for i in range(nodes_num)}
    # print(labels)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    nx.draw(G, pos=pos, node_color='y', edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.Reds_r,
            node_size=500)
    plt.savefig(str(path) + 'DAG_values' + str(iteration_number) + '.png')
    plt.show()
    #plt.figure()

def set_values_beta(adjacency_matrix, order, alpha_beta, theta, nonlinear_probability, b, a):
    nodes_num = len(adjacency_matrix)
    node_values = [np.random.beta(alpha_beta[0], alpha_beta[1]) for i in range(nodes_num)]
    for node in order:
        parents = get_linear_parents(adjacency_matrix, node)
        for par in parents:
            node_values[node] += node_values[par] * theta
        parents = get_nonlinear_parents(adjacency_matrix, node)
        for par in parents:
            node_values[node] += a * math.tanh(node_values[par] * b)
    return node_values

def simulate_beta(sparsity, y_has_children, n, observation_num, hidden_distance_bound,
             hide_probability, nonlinear_probability, a, b, theta, alpha_beta, path, simulation_num):
    '''
    Simulate Random Causal DAGs with values of noise follwing beta distribution
    '''
    graph_matrix = [[0 for _ in range(n)] for _ in range(n)]

    if y_has_children:
        topological_order = np.random.permutation(n)
    else:
        topological_order = np.append((np.random.permutation(n - 1) + 1), 0)
    for i, node1 in enumerate(topological_order):
        for node2 in topological_order[:i]:
            if np.random.uniform(0, 1) < sparsity:
                if np.random.uniform(0, 1) < nonlinear_probability:
                    graph_matrix[node2][node1] = 2
                else:
                    graph_matrix[node2][node1] = 1
            else:
                graph_matrix[node2][node1] = 0

    # print(topological_order)
    print(graph_matrix)
    # plot_graph(graph_matrix, path)
    node_names = ["Y"] + ["X" + str(i + 1) for i in range(n - 1)]
    print("node_names= " + str(node_names))
    true_graph = pd.DataFrame(graph_matrix, index=node_names, columns=node_names)
    true_graph.to_csv(str(path) + '/all_graph_matrix' + str(simulation_num) + '.csv', sep='\t')

    # hide some nodes based on the distance to target variable bounds and hide probability
    hidden_indices = list()
    # distances = find_node_distances_to_target_var(graph_matrix)
    # lower_bound, upper_bound = hidden_distance_bound
    # for i, dist in enumerate(distances[1:]):
    #     if (lower_bound <= dist <= upper_bound or dist == -1) and np.random.uniform(0, 1) < hide_probability:
    #         hidden_indices.append(i)
    # print("hidden indices are:" + str(["X" + str(i) for i in hidden_indices]))

    # for simulation in range(simulation_num):
    X = list()
    Y = list()
    for i in range(observation_num):
        values = set_values_beta(graph_matrix, topological_order, alpha_beta, theta, nonlinear_probability, b, a)
        #print(values)
        Y.append(values[0])
        X.append(values[1:])
        '''plot_graph_values(graph_matrix, values, simulation, i,  path)'''

    X, Y = np.asarray(X), np.asarray(Y)

    # save data to csv files
    simulated_data = {"X" + str(i + 1): X[:, i] for i in range(n - 1) if (i + 1) not in hidden_indices}
    simulated_data["Y"] = Y
    simulated_data = pd.DataFrame(simulated_data)
    simulated_data.to_csv(str(path) + '/simulated_data' + str(simulation_num) + '.csv', sep='\t')

    return true_graph

def simulate_normal(sparsity, y_has_children, n, observation_num, hidden_distance_bound,
             hide_probability, nonlinear_probability, a, b, theta, noise_sigma, path, simulation_num):
    '''
    Simulate Random Causal DAGs with values of noise follwing normal distribution
    '''
    graph_matrix = [[0 for _ in range(n)] for _ in range(n)]

    if y_has_children:
        topological_order = np.random.permutation(n)
    else:
        topological_order = np.append((np.random.permutation(n - 1) + 1), 0)
    for i, node1 in enumerate(topological_order):
        for node2 in topological_order[:i]:
            if np.random.uniform(0, 1) < sparsity:
                if np.random.uniform(0, 1) < nonlinear_probability:
                    graph_matrix[node2][node1] = 2
                else:
                    graph_matrix[node2][node1] = 1
            else:
                graph_matrix[node2][node1] = 0

    # print(topological_order)
    print(graph_matrix)
    # plot_graph(graph_matrix, path)
    node_names = ["Y"] + ["X" + str(i + 1) for i in range(n - 1)]
    print("node_names= " + str(node_names))
    true_graph = pd.DataFrame(graph_matrix, index=node_names, columns=node_names)
    true_graph.to_csv(str(path) + '/all_graph_matrix' + str(simulation_num) + '.csv', sep='\t')

    # hide some nodes based on the distance to target variable bounds and hide probability
    hidden_indices = list()
    # distances = find_node_distances_to_target_var(graph_matrix)
    # lower_bound, upper_bound = hidden_distance_bound
    # for i, dist in enumerate(distances[1:]):
    #     if (lower_bound <= dist <= upper_bound or dist == -1) and np.random.uniform(0, 1) < hide_probability:
    #         hidden_indices.append(i)
    # print("hidden indices are:" + str(["X" + str(i) for i in hidden_indices]))

    # for simulation in range(simulation_num):
    X = list()
    Y = list()
    for i in range(observation_num):
        values = set_values(graph_matrix, topological_order, 0, noise_sigma, theta, nonlinear_probability, b, a)
        #print(values)
        Y.append(values[0])
        X.append(values[1:])
        '''plot_graph_values(graph_matrix, values, simulation, i,  path)'''

    X, Y = np.asarray(X), np.asarray(Y)

    # save data to csv files
    simulated_data = {"X" + str(i + 1): X[:, i] for i in range(n - 1) if (i + 1) not in hidden_indices}
    simulated_data["Y"] = Y
    simulated_data = pd.DataFrame(simulated_data)
    simulated_data.to_csv(str(path) + '/simulated_data' + str(simulation_num) + '.csv', sep='\t')

    return true_graph


def find_node_distances_to_target_var(adjacency_matrix):
    queue = list()
    nodes_num = len(adjacency_matrix)
    distances = [-1 for _ in range(nodes_num)]
    distances[0] = 0
    queue.append(0)
    while len(queue) != 0:
        node = queue.pop()
        for neighbour in get_parents(adjacency_matrix, node):
            if distances[neighbour] == -1:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances  # contains nodes distances to target, -1 means there is no path


try:
    # Create target Directory
    os.mkdir("DAG Samples")
    print("Directory ", "DAG Samples",  " Created ")
except FileExistsError:
    print("Directory ", "DAG Samples",  " already exists")


alpha_beta = (0.5, 0.5)
nonlinear_probability = 0.3
observation_num = 100
n = 5
sparsity = 0.2
iteration_num = 10
a = 0.5
b = 1.5
if n == 20 or n == 50 or n >= 100:
    theta = 0.5
else:
    theta = 2

print("observation_num =" + str(observation_num))
print("n =" + str(n))
print("sparsity =" + str(sparsity))
print("nonlinear_probability =" + str(nonlinear_probability))
print("alpha_beta =" + str(alpha_beta))
print("--------------------")


path = "DAG Samples/Random Structures/" \
+ "sparsity(" + str(sparsity) + ")" \
+ "n(" + str(n) + ")" \
+ "observation_num(" + str(observation_num) + ")"\
+ "nonlinear_probability(" + str(nonlinear_probability) + ")"\
+ "a(" + str(a) + ")"\
+ "b(" + str(b) + ")"\
+ "theta(" + str(theta) + ")"\
+ "alpha(" + str(alpha_beta[0]) + ")" \
+ "beta(" + str(alpha_beta[1]) + ")"
try:
    # Create target Directory
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")
simulate_beta(sparsity=sparsity,
        y_has_children=False,
        n=n + 1,
        observation_num=int(observation_num),
        hidden_distance_bound=(2, 3),
        hide_probability=0,
        nonlinear_probability=nonlinear_probability,
        a=a,
        b=b,
        theta=theta,
        alpha_beta=alpha_beta,
        path=path,
        simulation_num=iteration_num)











