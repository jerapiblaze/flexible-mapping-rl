import numpy as np
import gymnasium as gym
import graph_mapping
import networkx as nx
import gzip
import pickle
import graph_generator
import graph_mapping_envs.static_mapping as static_mapping


def Main():
    PHY = graph_generator.phy.FromGml(
        path="./data/atlanta.gml", node_cap=[10, 100, 1], link_cap=[10, 100, 1])
    nx.draw_networkx(PHY)
    SFC_LIST = graph_generator.flex_sfc_set.PfoSFCSET(sfc_count=2, node_count_params=[
                                                      5, 6, 5], node_req_params=[10, 100, 1], link_req_params=[10, 100, 1], flex_rate_params=0)
    
    env = static_mapping.StaticMappingEnv(physical_graph=PHY, sfcs_list=SFC_LIST, key_attrs={"node_req":"weight", "link_req":"weight", "node_cap":"weight", "link_cap":"weight"})
    obs, info = env.reset()
    print("init", list(env.action_space), env.action_space[0])

    obs, reward, terminated, truncated, info = env.step(
        env.action_space[0])
    print("1st", list(env.action_space), env.action_space[0])
    print(terminated, truncated)

    obs, reward, terminated, truncated, info = env.step(
        env.action_space[0])
    print("2nd", list(env.action_space), env.action_space[0])
    print(terminated, truncated)


if __name__ == "__main__":
    Main()
    exit()
