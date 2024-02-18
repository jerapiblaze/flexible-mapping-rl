import networkx as nx
import random
import uuid

def PfoSFC(node_count_params, node_req_params, link_req_params, flex_rate_params: float = -1, seed=None):
    random.seed(seed)
    node_count = 5
    flex_rate = 0
    sfc = nx.DiGraph()

    sfc.name = f"pfosfc@{uuid.uuid4()}"
    sfc.attr = {
        "node_count": node_count,
        "flex_rate": flex_rate
    }

    nodes = []
    flex_links = {}

    for i in range(node_count):
        node_req = random.randint(
            node_req_params[0], node_req_params[1])
        link_req = random.randint(
            link_req_params[0], link_req_params[1])

        nodes.append((i, {"weight": node_req}))
        flex_links[i] = link_req

    sfc.add_nodes_from(nodes)

    links = []
    for i in range(node_count - 1):
        links.append((nodes[i][0], nodes[i+1][0],
                      {"weight": flex_links[nodes[i][0]]}))

    flex_link_count = int(round(len(links) * flex_rate, 0))

    for i in range(flex_link_count):
        link_to_remove = random.choice(links)
        links.remove(link_to_remove)

    sfc.add_edges_from(links)
    sfc.FlexEndpointRequirement = flex_links

    return sfc

def PfoSFCSET(sfc_count: int, node_count_params, node_req_params, link_req_params, flex_rate_params: float = -1, seed=None):
    SFC_SET = []

    node_count = 5

    for i in range(sfc_count):
        sfc = PfoSFC(node_count_params=node_count, node_req_params=node_req_params,
                     link_req_params=link_req_params, flex_rate_params=flex_rate_params, seed=seed)
        SFC_SET.append(sfc)

    return SFC_SET

if __name__ == "__main__":
    node_count_params = (5, 5, 1)  # Fix node_count to 5
    node_req_params = (5, 10)  # Node weight fluctuates between 5 and 10
    link_req_params = (20, 40)  # Link weight fluctuates between 20 and 40
    flex_rate = 0
    seed_value = 42  # Set a specific seed value

    SFC_SET = PfoSFCSET(2, node_count_params, node_req_params, link_req_params, flex_rate, seed=seed_value)
    
    for idx, sfc in enumerate(SFC_SET):
        
        # In thông tin về các nút
        print("\nNodes:")
        for node, data in sfc.nodes(data=True):
            print(f"Node {node}: {data}")

        # In thông tin về các cạnh và trọng số
        print("\nEdges:")
        for edge in sfc.edges(data=True):
            print(f"Edge {edge}")
        
        print("\n")
