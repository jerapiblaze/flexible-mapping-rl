import gymnasium as gym
import networkx as nx
import copy
import numpy as np

class StaticMapping2Env(gym.Env):
    # Actions space
    action_space = gym.Space()
    # States space
    observation_space = gym.Space()
    # Reward range
    reward_range = (-100, 100)
    # Mapping infomations
    # Original
    physical_graph = nx.DiGraph()
    sfcs_list = list()
    key_attrs = dict()
    vnf_order = list()
    # State
    physical_graph_current = nx.DiGraph()
    vnf_order_index_current = int()
    node_solution_current = list() # list[tuple[sfc_id:int, vnode_id:int, node_id:int]] 
    link_solution_current = list() # list[tuple[sfc_id:int, vlink_id:tuple[int,int], link_id:tuple[int,int]]]
    node_solution_lastgood = list()
    link_solution_lastgood = list()
    node_solution = list()
    link_solution = list()
    __mapped_sfc = int()

    def __init__(self, physical_graph: nx.DiGraph, sfcs_list: list[nx.DiGraph], key_attrs: dict[str:str]):
        self.physical_graph = copy.deepcopy(physical_graph)
        self.sfcs_list = copy.deepcopy(sfcs_list)
        self.key_attrs = copy.deepcopy(key_attrs)
        self.vnf_order = VNodeMappingOrderCompose(self.sfcs_list)
        # Observation space: {0, 1} -> True if a node is mapped
        self.observation_space = gym.spaces.Discrete(n=2, seed=42, start=0)
        # Action space: {0, 1, 2, ..., n} -> physical node id
        self.action_space = gym.Space(list(self.physical_graph.nodes), dtype='int64')

    def reset(self, seed=None, options=None):
        # Initialize state
        self.physical_graph_current = copy.deepcopy(self.physical_graph)
        self.vnf_order_index_current = 0
        self.node_solution_current  = list()
        self.link_solution_current  = list()
        self.node_solution_lastgood = list()
        self.link_solution_lastgood = list()
        self.node_solution = list()
        self.link_solution = list()
        self.__mapped_sfc = 0
        return (False, {"message": "environment reset"})

    def __execute_node_mapping(self, sfc_id, vnf_id, node_id):
        vnode_req = self.__get_vnode_req(sfc_id, vnf_id)
        # print(f"vnode {vnf_id} _req:", vnode_req)
        nodes_cap = self.__get_node_cap(None)
        nodes_cap[node_id] -= vnode_req
        # print(f"node_cap{node_id}: ",nodes_cap[node_id])
        if any(node < 0 for node in nodes_cap):
            raise nx.NetworkXUnfeasible(f"Requested vnode sfc={sfc_id} vnf={vnf_id} has exceed capacity of node={node_id}")
        nx.set_node_attributes(self.physical_graph_current, nodes_cap, name="weight")
        self.node_solution_current.append((sfc_id, vnf_id, node_id))
  

    def __execute_link_mapping(self, sfc_id, vlink_id, link_id):
        vlink_req = self.__get_vlink_req(sfc_id, vlink_id)
        links_cap = self.__get_link_cap(None)
        links_cap[link_id] -= vlink_req
        if any(link < 0 for link in links_cap.values()):
            raise nx.NetworkXUnfeasible(f"Requested vnode sfc={sfc_id} vnf={vlink_id} has exceed capacity of node={link_id}")
        nx.set_edge_attributes(self.physical_graph_current, links_cap, name="weight")
        self.link_solution_current.append((sfc_id, vlink_id, link_id))
        
    def __get_action_details(self, action):
        node_id = action
        sfc_id, vnf_id = self.vnf_order[self.vnf_order_index_current]
        sfc_id_prev, vnf_id_prev = self.vnf_order[self.vnf_order_index_current - 1]
        search_result = [node_sol[2] for node_sol in self.node_solution_current if node_sol[0] == sfc_id_prev and node_sol[1] == vnf_id_prev]
        node_id_prev = search_result[0] if len(search_result) else None
        return node_id, sfc_id, vnf_id, node_id_prev, sfc_id_prev, vnf_id_prev
    
    def __is_first_of_sfc(self):
        if (self.__is_terminated()):
            return False
        if (self.vnf_order_index_current == 0):
            return True
        sfc_id, vnf_id = self.vnf_order[self.vnf_order_index_current]
        sfc_id_prev, vnf_id_prev = self.vnf_order[self.vnf_order_index_current - 1]
        if sfc_id == sfc_id_prev:
            return False
        return True
    
    def __is_last_of_sfc(self):
        if (self.__is_terminated()):
            return True
        if ((self.vnf_order_index_current + 1) >= len(self.vnf_order)):
            return True
        sfc_id, vnf_id = self.vnf_order[self.vnf_order_index_current]
        sfc_id_next, vnf_id_next = self.vnf_order[self.vnf_order_index_current + 1]
        if sfc_id == sfc_id_next:
            return False
        return True
    
    def __get_node_cap(self, node_id):
        node_caps = nx.get_node_attributes(self.physical_graph_current, name="weight")
        if (node_id is None):
            return node_caps
        return node_caps[node_id]
    
    def __get_link_cap(self, link_id):
        link_caps = nx.get_edge_attributes(self.physical_graph_current, name="weight")
        if (link_id is None):
            return link_caps
        return link_caps[link_id]   

    def __get_vnode_req(self, sfc_id, vnf_id): #lay capa của vnf
        vnf_reqs = nx.get_node_attributes(self.sfcs_list[sfc_id], name="weight")
        if (vnf_id is None):
            return vnf_reqs
        return vnf_reqs[vnf_id]
    
    def __get_vlink_req(self, sfc_id, vlink_id):
        vlink_reqs = nx.get_edge_attributes(self.sfcs_list[sfc_id], name="weight")
        if (vlink_id is None):
            return vlink_reqs
        return vlink_reqs[vlink_id]

    def __validate_action(self, action):
        node_id, sfc_id, vnf_id, node_id_prev, sfc_id_prev, vnf_id_prev = self.__get_action_details(action)
        # Validate node capacity
        node_cap = self.__get_node_cap(node_id)
        # print(node_cap)
        vnf_req = self.__get_vnode_req(sfc_id, vnf_id)
        if vnf_req > node_cap:
            # print(f"vnf_req: {vnf_req} - node-cap: {node_cap}")  
            reward = -100
            # info = {
            #     "message": f"Requirement of {sfc_id}_{vnf_id} beyound capacity of {node_id}"
            # }
            return f"Requirement of {sfc_id}_{vnf_id} beyound capacity of {node_id}"
            # return (False, reward, self.__is_terminated(), self.__is_truncated(), info)
        # Validate node singuality
        # If first node, no need to check
        if self.__is_first_of_sfc():
            return None
        # Check if node is used or not
        if any(node_sol[0] == sfc_id and node_sol[2] == node_id for node_sol in self.node_solution_current):

            # reward = -100
            # info = {
            #     "message": f"Node {node_id} is already used"
            # }
            # return (False, reward, self.__is_terminated(), self.__is_truncated(), info)            
            return f"Node {node_id} is already used"

        return None
    
    def __is_terminated(self):
        if (self.vnf_order_index_current >= len(self.vnf_order)):
            return True
        return False
    
    def __is_truncated(self):
        return False
    
    def is_full_mapping(self):
        if (self.__mapped_sfc == len(self.sfcs_list)):
            return True
        return False

    # TODO: SKIP SFC
    def __skip_sfc(self):
        while(True):
            if self.__is_terminated():
                return
            if (self.vnf_order_index_current == 0):
                self.vnf_order_index_current += 1
                continue
            vnf_order = self.vnf_order[self.vnf_order_index_current]
            vnf_order_prev = self.vnf_order[self.vnf_order_index_current - 1]
            sfc_id, vnf_id = vnf_order
            sfc_id_prev, vnf_id_prev = vnf_order_prev
            if (sfc_id == sfc_id_prev):
                self.vnf_order_index_current += 1
            else:
                return
    
    def __confirm_mapping(self):
        self.node_solution_lastgood = copy.deepcopy(self.node_solution_current)
        
        self.link_solution_lastgood = copy.deepcopy(self.link_solution_current)
        self.vnf_order_index_current += 1

    def __abort_mapping(self): # hủy mapping
        self.node_solution_current = copy.deepcopy(self.node_solution_lastgood)
        self.link_solution_current = copy.deepcopy(self.link_solution_lastgood)

    def __confirm_solution(self):
        self.node_solution = copy.deepcopy(self.node_solution_lastgood)
        self.link_solution = copy.deepcopy(self.link_solution_lastgood)
        self.__mapped_sfc += 1

    # def __confirm_solution2(self):
    #     self.node_solution = copy.deepcopy(self.node_solution_lastgood)
    #     self.link_solution = copy.deepcopy(self.link_solution_lastgood)

    def step(self, action):
        # print("cu rent: ", self.node_solution_current)
        # Skip the sfc action
        if (action == -1):
            self.__skip_sfc()
            reward = 0
            info = {
                "message": "skip the sfc"
            }
            return (False, reward, self.__is_terminated(), self.__is_truncated(), info)
        
        # If terminated or failed earlier, do nothing
        if (self.__is_terminated() or self.__is_truncated()):
            reward = 0
            info = {
                "message": "the env is terminated or truncated"
            }
            return (False, reward, self.__is_terminated(), self.__is_truncated(), info)
        
        # Check if first action and last action
        is_first = self.__is_first_of_sfc()
        is_last = self.__is_last_of_sfc()
        
        reward = 0
        info = {}
        # If action is invalid
        action_validation = self.__validate_action(action)
        if (action_validation):
            reward = -100
            info = {
                "message": f"action invalid: {action_validation}"
            }
            # self.__confirm_solution2()
            return (False, reward, True, self.__is_truncated(), info)

        node_id, sfc_id, vnf_id, node_id_prev, sfc_id_prev, vnf_id_prev = self.__get_action_details(action)
        # print(f"action detail: node_id: {node_id}, sfc_id: {sfc_id}, vnf_id:  {vnf_id}, node_id_prev: {node_id_prev}, sfc_id_prev: {sfc_id_prev}, vnf_id_prev: {vnf_id_prev}")
        ai_t = self.__get_node_cap(node_id)
        rv = self.__get_vnode_req(sfc_id, vnf_id)
        M = 200  # Replace with your desired constant
        beta = 5 
        # First, try to map node
        try:
            self.__execute_node_mapping(sfc_id, vnf_id, node_id) # mapping
        except nx.NetworkXUnfeasible: # nếu k được
            self.__abort_mapping()  # hủy mapping - lưu lại cái tốt nhất trc đó
            info = {
                "message": f"no node for {sfc_id}_{vnf_id} ({node_id})"
            }
            reward = -1
            return (False, reward, self.__is_terminated(), self.__is_truncated(), info)
        # If is the first action of a sfc, no need to map link
        if is_first:
            self.__confirm_mapping()
            # reward = 1
            reward_first_vnf = M - (ai_t - rv)
            reward += reward_first_vnf
            info = {
                "message": "first action success"
            }
            return (True, reward, self.__is_terminated(), self.__is_truncated(), info)
        
        # Normal action, try to map link
        try:
            vlink = (vnf_id_prev, vnf_id)
            # print(f"vlink {vnf_id_prev} {vnf_id}", vlink)
            vlink_req = self.__get_vlink_req(sfc_id, vlink)
            # print("vlink_req: ", vlink_req)
            paths = PhysicalNodeConnect(self.physical_graph_current, node_id_prev, node_id, vlink_req)
            # print(paths)
            paths = GetPathListFromPath(paths)
            for path in paths:
                self.__execute_link_mapping(sfc_id, vlink, path)
        except nx.NetworkXUnfeasible:
            self.__abort_mapping()
            info = {
                "message": f"no link for {sfc_id_prev}_{vnf_id_prev}-{sfc_id}_{vnf_id} ({node_id_prev}-{node_id})"
            }
            reward = -1
            return (False, reward, self.__is_terminated(), self.__is_truncated(), info)
        self.__confirm_mapping()
        # Tính số lượng bước nhảy (hops) giữa nút hiện tại và nút trước đó
        if node_id_prev is not None:
            hop_count = len(nx.shortest_path(self.physical_graph_current, source=node_id_prev, target=node_id)) - 1
        else:
            hop_count = 0

        reward_normal_action = M - (ai_t - rv) - beta * hop_count
        reward += reward_normal_action
        info = {
            "message": f"action success"
        }
        # If last action success, the sfc is success
        if is_last:
            self.__confirm_solution()
            info = {
                "message": "sfc ok"
            }
        return (True, reward, self.__is_terminated(), self.__is_truncated(), info)

    def render(self):
        pass

    def close(self):
        pass

def GetPathListFromPath(path):
    return [(a, b) for a in path for b in path if path.index(b)-path.index(a) == 1]

def VNodeMappingOrderCompose(sfcs_list: list[nx.DiGraph]):
    order = []
    for i in range(len(sfcs_list)):
        sfc = sfcs_list[i]
        for vnode in sfc.nodes:
            order.append((i, vnode))
    return order

def PhysicalNodeConnect(graph, start, end, requirement):
    tmp_graph = nx.restricted_view(
        graph,
        [],
        tuple((x, y) for x, y, attr in graph.edges(data=True) if attr["weight"] <= requirement)
    )
    path = nx.shortest_path(tmp_graph, start, end, requirement)
    return path