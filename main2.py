import numpy as np
import gymnasium as gym
import graph_mapping
import networkx as nx
import gzip
import pickle
import graph_generator
import random

def Main():
    PHY = graph_generator.phy.FromGml(
        path="./data/atlanta.gml", node_cap=[10, 100, 1], link_cap=[10, 100, 1])
    nx.draw_networkx(PHY)
    SFC_LIST = graph_generator.flex_sfc_set.PfoSFCSET(sfc_count=5, node_count_params=[
                                                      5, 6, 5], node_req_params=[10, 50, 1], link_req_params=[10, 50, 1], flex_rate_params=0)
    env = gym.make(id="graph_mapping/static_mapping-v2", physical_graph=PHY, sfcs_list=SFC_LIST,
                   key_attrs={"node_req": "weight", "link_req": "weight", "node_cap": "weight", "link_cap": "weight"})

    for i in range(1):
        print(f"EPISOE: {i}")
        obs, info = env.reset()
        print(obs, info)
        print(env.action_space.shape)
        print(env.observation_space.shape)
        terminated = False
        truncated = False
        j = 0
        # SUPER AGENT
        while (not terminated and not truncated):
            action = env.action_space.shape[j] if (not j == -1) else -1
            print("state", env.vnf_order[env.vnf_order_index_current])
            print("action",action)
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward, terminated, truncated, info)
            if (terminated):
                if (env.is_full_mapping()):
                    print("mapping success", env.node_solution, env.link_solution)
                else:
                    print("mapping partially", env.node_solution, env.link_solution)
            if action == -1:
                j = 0
                continue
            if not obs:
                j += 1
            else:
                j = 0
            if j >= len(env.action_space.shape):
                j = -1

if __name__ == "__main__":
    Main()
    exit()
