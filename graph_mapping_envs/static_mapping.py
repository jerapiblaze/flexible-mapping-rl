import gymnasium as gym
import networkx as nx
import copy
import numpy as np

class StaticMappingEnv():
    # Actions space
    action_space = list()
    # Reward range
    reward_range = (0, 0)
    # Mapping infomations
    # Original
    physical_graph = nx.DiGraph()
    sfcs_list = list()
    key_attrs = dict()
    # Current
    physical_graph_current = nx.DiGraph()
    vnode_order_current = list()
    action_history = list()
    terminated = False
    truncated = False

    def __init__(self, physical_graph: nx.DiGraph, sfcs_list: list[nx.DiGraph], key_attrs: dict[str:str]):
        # def Create(self, physical_graph:nx.DiGraph, sfcs_list:list[nx.DiGraph], key_attrs:dict[str:str]):
        self.physical_graph = copy.deepcopy(physical_graph)
        self.sfcs_list = copy.deepcopy(sfcs_list)
        self.key_attrs = copy.deepcopy(key_attrs)

    def reset(self, seed=None, options=None):
        self.physical_graph_current = copy.deepcopy(self.physical_graph)
        self.vnode_order_current = VNodeMappingOrderCompose(self.sfcs_list)
        self.action_history = list()
        self.__calc_action_space()
        return (True, {"message": "environment reset"})

    def __calc_action_space(self):
        if not len(self.vnode_order_current):
            self.terminated = True
            return None
        vnode = self.vnode_order_current.pop()
        actions = []
        vnode_req = nx.get_node_attributes(
            self.sfcs_list[vnode[0]], name="weight")[vnode[1]]
        node_caps = nx.get_node_attributes(
            self.physical_graph_current, name="weight")
        for node in self.physical_graph_current.nodes:
            skip_node = False
            for action_his in self.action_history:
                sfc_id_his, vnf_id_his, node_id_his = ActionParser(action_his)
                if node_id_his == node:
                    skip_node = True
                    break
            if skip_node:
                continue
            node_cap = node_caps[node]
            if node_cap >= vnode_req:
                actions.append((vnode[0], vnode[1], node))
        self.action_space = actions
        return self.action_space

    def __execute_node_mapping(self, sfc_id, vnf_id, node_id):
        vnode_req = nx.get_node_attributes(
            self.sfcs_list[sfc_id], name="weight")[vnf_id]
        nodes_cap = nx.get_node_attributes(
            self.physical_graph_current, name="weight")
        nodes_cap[node_id] -= vnode_req
        if any(node < 0 for node in nodes_cap):
            raise Exception(f"Requested vnode sfc={sfc_id} vnf={vnf_id} has exceed capacity of node={node_id}")
        nx.set_node_attributes(
            self.physical_graph_current, nodes_cap, name="weight")

    def __execute_link_mapping(self, sfc_id, vlink_id, link_id):
        vlink_req = nx.get_edge_attributes(
            self.sfcs_list[sfc_id], name="weight")[vlink_id]
        links_cap = nx.get_edge_attributes(
            self.physical_graph_current, name="weight")
        links_cap[link_id] -= vlink_req
        if any(link < 0 for link in links_cap):
            raise Exception(f"Requested vnode sfc={sfc_id} vnf={vlink_id} has exceed capacity of node={link_id}")
        nx.set_edge_attributes(
            self.physical_graph_current, links_cap, name="weight")

    def step(self, action):
        # print("action", action)
        # print("order", self.vnode_order_current)
        reward = 0
        info = {}
        if (action not in self.action_space):
            reward = -1
            self.terminated = True
            self.truncated = True
            info = {
                "message": "the requested action is not in action space"
            }
            return (False, reward, self.terminated, self.truncated, info)

        sfc_id, vnf_id, node_id = ActionParser(action)
        # First action of sfc
        is_first_action = True
        for action_his in self.action_history:
            sfc_id_his, vnf_id_his, node_id_his = ActionParser(action_his)
            if sfc_id_his == sfc_id:
                is_first_action = False
                break
        if is_first_action:
            self.__execute_node_mapping(sfc_id, vnf_id, node_id)
            self.__calc_action_space()
            self.action_history.append(action)
            reward = 1
            info = {
                "message": "first action"
            }
            return (True, reward, self.terminated, self.truncated, info)

        # Normal action
        last_action = self.action_history[-1]
        sfc_id_prev, vnf_id_prev, node_id_prev = ActionParser(last_action)
        try:
            link_req = nx.get_edge_attributes(self.sfcs_list[sfc_id], name="weight")[
                (vnf_id_prev, vnf_id)]
            paths = PhysicalNodeConnect(
                self.physical_graph_current, node_id_prev, node_id, link_req)
            print("PATH", paths)
        except nx.NetworkXUnfeasible:
            self.terminated = True
            self.truncated = True
            info = {
                "message": f"no link for {sfc_id_prev}_{vnf_id_prev}-{sfc_id}_{vnf_id} ({node_id_prev}-{node_id})"
            }
            reward = -1
            return (False, reward, self.terminated, self.truncated, info)
        self.action_history.append(action)
        return (True, reward, self.terminated, self.truncated, info)

    def render(self):
        pass

    def close(self):
        pass


def VNodeMappingOrderCompose(sfcs_list: list[nx.DiGraph]):
    order = []
    for i in range(len(sfcs_list)):
        sfc = sfcs_list[i]
        for vnode in sfc.nodes:
            order.append((i, vnode))
    order.reverse()
    return order


def ActionParser(action: tuple[int, int, int]):
    sfc_id = action[0]
    vnf_id = action[1]
    node_id = action[2]
    return sfc_id, vnf_id, node_id


def PhysicalNodeConnect(graph, start, end, requirement):
    # def weight(u,v,attr):
    #     if (attr["weight"] < requirement):
    #         return None
    #     return 1
    # return nx.dijkstra_path(graph, start, end, weight)
    tmp_graph = nx.restricted_view(
        graph,
        [],
        tuple((x, y) for x, y, attr in graph.edges(
            data=True) if attr["weight"] <= requirement)
    )
    path = nx.shortest_path(tmp_graph, start, end, requirement)
    return path
