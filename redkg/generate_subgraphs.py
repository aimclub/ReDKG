import random
import torch
from torch_geometric.data import Data


def generate_subgraphs(dataset, num_subgraphs=5, min_nodes=2, max_nodes=5):
    subgraphs = []
    for _ in range(num_subgraphs):
        selected_nodes = []
        while len(selected_nodes) < random.randint(min_nodes, max_nodes):
            if selected_nodes:
                new_node = random.choice(
                    [link['target'] for link in dataset['links'] if
                     link['source'] in {node['id'] for node in selected_nodes}] +
                    [link['source'] for link in dataset['links'] if
                     link['target'] in {node['id'] for node in selected_nodes}]
                )
            else:
                new_node = random.choice(dataset['nodes'])['id']
            if new_node not in {node['id'] for node in selected_nodes}:
                selected_nodes.append({'id': new_node})
        selected_node_ids = {node['id'] for node in selected_nodes}
        selected_links = [link for link in dataset['links'] if
                          link['source'] in selected_node_ids and link['target'] in selected_node_ids]
        subgraphs.append({'nodes': selected_nodes, 'links': selected_links})
    return subgraphs


def generate_subgraphs_dataset(subgraphs, large_dataset):
    # Generate dataset from all subgraphs
    dataset = []
    for i in range(len(subgraphs)):
        user_edge_index = []
        for link in subgraphs[i]['links']:
            source_idx = large_dataset.node_mapping.get(link['source'])
            target_idx = large_dataset.node_mapping.get(link['target'])
            # Add edge only if both nodes are on the subgraph
            if source_idx is not None and target_idx is not None:
                user_edge_index.append([source_idx, target_idx])
        user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()

        # Convert subgraphs nodes of the small graph
        user_node_index = []
        for link in subgraphs[i]['nodes']:
            node_idx = large_dataset.node_mapping.get(link['id'])
            if node_idx is not None:
                user_node_index.append(node_idx)
        user_node_indices = large_dataset.x[user_node_index]

        # Make a mask for the subgraph nodes
        user_mask = torch.zeros_like(large_dataset.x)
        for idx in user_node_index:
            user_mask[idx] = 1
        masked_features = large_dataset.x * user_mask

        # Create a dataset from the subgraph using the same features and labels as the original dataset
        user_data = Data(x=masked_features, edge_index=user_edge_index, y=large_dataset.y)

        dataset.append(user_data)

    return dataset


if __name__ == "__main__":
    graph = {
        "nodes": [
            {"id": "vb.net"},
            {"id": "vb"},
            {"id": "net"},
            {"id": "assembly"},
            {"id": "mvc"},
            {"id": "python"},
            {"id": "linq"},
            {"id": "html"},
            {"id": "sql"}
        ],
        "links": [
            {"source": "assembly", "target": "html"},
            {"source": "assembly", "target": "net"},
            {"source": "assembly", "target": "python"},
            {"source": "assembly", "target": "sql"},
            {"source": "html", "target": "net"},
            {"source": "html", "target": "python"},
            {"source": "html", "target": "sql"},
            {"source": "linq", "target": "mvc"},
            {"source": "linq", "target": "net"},
            {"source": "mvc", "target": "net"},
            {"source": "mvc", "target": "sql"},
            {"source": "net", "target": "python"},
            {"source": "net", "target": "sql"},
            {"source": "net", "target": "vb"},
            {"source": "net", "target": "vb.net"},
            {"source": "vb", "target": "vb.net"}
        ]
    }
    combined_graph = generate_subgraphs(graph)
    import json

    print(json.dumps(combined_graph))
