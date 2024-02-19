# chat GPT

import numpy as np
import gymnasium as gym
import graph_mapping
import networkx as nx
import gzip
import pickle
import graph_generator
import graph_generator.sfc_set2 as sfc_set
import graph_mapping.envs.static_mapping2 as static_mapping
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(0, 1, size=(state_space_size, action_space_size))

    def choose_action(self, state):
        rand = np.random.rand()
        if state == True:
            cr_s = 1
        else: 
            cr_s = 0        
        if rand < self.epsilon:
            a = np.random.choice(self.action_space_size)
            # print("a = ",a)
            return a
            
        else:
            # print("max = ",np.argmax(self.q_table[cr_s]))
            return np.argmax(self.q_table[cr_s])

    def update_q_table(self, state, action, reward, next_state):
        if state == True:
            cr_s = 1
        else: 
            cr_s = 0 
        if next_state == True:
            n_s = 1
        else: 
            n_s = 0        
        best_next_action = np.argmax(self.q_table[n_s])
        # print("curent q: ", self.q_table[cr_s, action])
        self.q_table[cr_s, action] += self.alpha * (reward + self.gamma * self.q_table[n_s, best_next_action] - self.q_table[cr_s, action])
        # print("new q: ", self.q_table[cr_s, action] )

random_seed = 42 
PHY = nx.DiGraph()
nodes = range(0, 7)
for node in nodes:
    PHY.add_node(node, weight=100)
max_edges = 15
edges = [(i, j) for i in range(1, 7) for j in range(1, 7) if i != j]
edges = edges[:max_edges] 
for edge in edges:
    PHY.add_edge(*edge, weight=100)
    PHY.add_edge(edge[1], edge[0], weight=100)

# print("\nNode Weights:")
# node_weights = nx.get_node_attributes(PHY, "weight")
# for node, weight in node_weights.items():
#     print(f"Node {node}: {weight}")

# # In trọng số của các cạnh
# print("\nEdge Weights:")
# edge_weights = nx.get_edge_attributes(PHY, "weight")
# for edge, weight in edge_weights.items():
#     print(f"Edge {edge}: {weight}")      

SFC_LIST = graph_generator.flex_sfc_set2.PfoSFCSET(sfc_count=2, node_count_params=[5, 6, 5], node_req_params=[10, 50, 1], link_req_params=[10, 50, 1], flex_rate_params=0, seed=random_seed)
for idx, sfc in enumerate(SFC_LIST):
    
    # In thông tin về các nút
    print("\nNodes:")
    for node, data in sfc.nodes(data=True):
        print(f"Node {node}: {data}")
    # In thông tin về các cạnh và trọng số
    print("\nEdges:")
    for edge in sfc.edges(data=True):
        print(f"Edge {edge}")
    
    print("\n")    
# Create environment

env = static_mapping.StaticMapping2Env(physical_graph=PHY, sfcs_list=SFC_LIST, key_attrs={"node_req":"weight", "link_req":"weight", "node_cap":"weight", "link_cap":"weight"})


# Example usage
state_space_size = 2  # True or False
action_space_size = len(env.action_space.shape)
# print(action_space_size)
agent = QLearningAgent(state_space_size, action_space_size)

# with open("./qvalues.csv", "wt") as f:
#     f.write("episode,qvalue\n")

# Training loop
for episode in range(10000):
    print("eps:", episode)
    state, info = env.reset()
    terminated = False
    truncated = False
    
    while (not terminated and not truncated):
    # for _ in range(10):
        if terminated:
            print("Done")
        else: 
        
            # print("list of order: ", env.vnf_order)
            # print("order: ",env.vnf_order_index_current)
            # print("state: ", state)
            action = agent.choose_action(state)
            # print("action = ",action)
            # print(agent.q_table)
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(next_state, reward, terminated, truncated, info)
            # print("DONE?: ", terminated)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            if (terminated):
                if (env.is_full_mapping()):
                    print("mapping success", env.node_solution, env.link_solution)
                # else:
                    # print("mapping partially", env.node_solution, env.link_solution)
            # qvalue = np.sum(agent.q_table)
            if episode == 9999:
                print(agent.q_table)

    # with open("./qvalues.csv", "at") as f:
    #     f.write(f"{episode},{qvalue}\n")
# a = env._StaticMapping2Env__validate_action(3)
# print(a)
