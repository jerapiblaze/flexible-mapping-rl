import networkx as nx
import random
import uuid


def PfoSFC(node_count_params: tuple[int, int, int], node_req_params: tuple[int, int, int], link_req_params: tuple[int, int, int], flex_rate_params: float = -1):
    node_count = random.randrange(
        start=node_count_params[0], stop=node_count_params[1], step=node_count_params[2])
    flex_rate = random.random() if flex_rate_params < 0 else flex_rate_params

    sfc = nx.DiGraph()

    # Metadata
    sfc.name = f"pfosfc@{uuid.uuid4()}"
    sfc.attr = {
        "node_count": node_count,
        "node_req_min": node_req_params[0],
        "node_req_max": node_req_params[1],
        "node_req_step": node_req_params[2],
        "link_req_min": link_req_params[0],
        "link_req_max": link_req_params[1],
        "link_req_step": link_req_params[2],
        "flex_rate": flex_rate
    }

    # Node gen
    nodes = []
    flex_links = {}
    for i in range(node_count):
        node_req = random.randrange(
            start=node_req_params[0], stop=node_req_params[1], step=node_req_params[2])
        link_req = random.randrange(
            start=link_req_params[0], stop=link_req_params[1], step=link_req_params[2])
        nodes.append((i, {"weight": node_req}))
        flex_links[i] = link_req
    sfc.add_nodes_from(nodes)

    # Link gen
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


def PfoSFCSET(sfc_count: int, node_count_params, node_req_params, link_req_params, flex_rate_params: float = -1):
    SFC_SET = []
    # node_count = (node_count_params.Min, node_count_params.Max,
    #               node_count_params.Step)
    # node_req = (node_req_params.Min, node_req_params.Max, node_req_params.Step)
    # link_req = (node_req_params.Min, node_req_params.Max, node_req_params.Step)
    for i in range(sfc_count):
        sfc = PfoSFC(node_count_params, node_req_params,
                     link_req_params, flex_rate_params)
        SFC_SET.append(sfc)
    return SFC_SET


if __name__ == "__main__":
    node_count = [5, 7, 1]
    node_req = [10, 100, 5]
    link_req = [20, 80, 2]
    flex_rate = 0.5

    sfc = PfoSFC(node_count, node_req, link_req, flex_rate)

    # print(sfc.FlexEndpointRequirement[3]["weight"])
    pass
