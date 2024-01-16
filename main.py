import numpy as np
import gymnasium as gym
import graph_mapping
import networkx as nx
import gzip
import pickle
import graph_generator

def Main():
    PHY = graph_generator.phy.FromGml(path="./data/atlanta.gml", node_cap=[10,100,1],link_cap=[10,100,1])
    nx.draw_networkx(PHY)
    SFC_LIST = graph_generator.flex_sfc_set.PfoSFCSET(sfc_count=2, node_count_params=[5,6,5], node_req_params=[10,100,1], link_req_params=[10,100,1], flex_rate_params=0)
    env = gym.make(id="graph_mapping/static_mapping-v0", physical_graph=PHY, sfcs_list=SFC_LIST, key_attrs={"node_req":"weight", "link_req":"weight", "node_cap":"weight", "link_cap":"weight"})
    # env.Create(physical_graph=PHY, sfcs_list=SFC_LIST, key_attrs={"node_req":"weight", "link_req":"weight", "node_cap":"weight", "link_cap":"weight"})
    for i in range(10000):
        print(f"episode: {i}")
        obs, info = env.reset()
        if not obs:
            continue
        print("init", list(env.action_space.shape), info)
        while (len(env.action_space.shape) and (not env.truncated and not env.terminated)):
            obs, reward, terminated, truncated, info = env.step(env.action_space.shape[0])
            print("after_action",list(env.action_space.shape), info)

if __name__=="__main__":
    Main()
    exit()