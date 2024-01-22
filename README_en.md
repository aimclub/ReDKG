# ReDKG
[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat.svg)](https://en.itmo.ru/en/)
[![Documentation](https://github.com/aimclub/ReDKG/actions/workflows/gh_pages.yml/badge.svg)](https://aimclub.github.io/ReDKG/)
[![Linters](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml)
[![Tests](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml)
[![Mirror](https://img.shields.io/badge/mirror-GitLab-orange)](https://gitlab.actcognitive.org/itmo-sai-code/ReDKG)
<p align="center">
  <img src="https://github.com/aimclub/ReDKG/blob/main/docs/img/logo.png?raw=true" width="300px"> 
</p>


**Re**inforcement learning on **D**ynamic **K**nowledge **G**raphs (**ReDKG**) 
is a toolkit for deep reinforcement learning on dynamic knowledge graphs. 
It is designed to encode static and dynamic knowledge graphs (KG) by constructing vector representations for the entities and relationships. 
The reinforcement learning algorithm based on vector representations is designed to train recommendation models or models of decision support systems based on reinforcement learning (RL) using vector representations of graphs.


## Installation

Python >= 3.9 is required

As a first step, [Pytorch Geometric installation](https://github.com/pyg-team/pytorch_geometric/) and Torch 1.1.2 are required.

### PyTorch 1.12

```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

When Torch installed clone this repo and run inside repo directory:

```
pip install . 
```

## Download test data

Download [ratings.csv](https://grouplens.org/datasets/movielens/20m/) to /data/ folder./
Data folder should contain the following files: 

 - `ratings.csv` - raw rating file;
 - `attributes.csv` - raw attributes file;
 - `kg.txt` - knowledge graph file;
 - `item_index2enity_id.txt` - the mapping from item indices in the raw rating file to entity IDs in the KG file;

## Example of Using KGE Models

### Preprocess the data

```python
from redkg.config import Config
from redkg.preprocess import DataPreprocessor

config = Config()
preprocessor = DataPreprocessor(config)
preprocessor.process_data()
```

### Train KG model

```python
kge_model = KGEModel(
    model_name="TransE",
    nentity=info['nentity'],
    nrelation=info['nrelation'],
    hidden_dim=128,
    gamma=12.0,
    double_entity_embedding=True,
    double_relation_embedding=True,
    evaluator=evaluator
)

training_logs, test_logs = train_kge_model(kge_model, train_pars, info, train_triples, valid_triples)
```

## Example of Using GCN, GAT, GraphSAGE Models

These models implement an algorithm for predicting links in a knowledge graph.

Additional information about training steps can be found in [basic_link_prediction.ipynb](https://github.com/aimclub/ReDKG/blob/main/examples/basic_link_prediction.ipynb) example.

### Loading Test Data

The test dataset can be obtained from the link [jd_data2.json](https://github.com/ZhongTr0n/JD_Analysis/blob/main/jd_data2.json)
and placed in the `/data/` directory.

### Data Preprocessing

For preprocessing, it is necessary to read data from the file and convert it into PyTorch Geometric format.

```python
import json
import torch
from torch_geometric.data import Data

# Read data from the file
with open('jd_data2.json', 'r') as f:
    graph_data = json.load(f)

# Extract the list of nodes and convert it to a dictionary for quick lookup
node_list = [node['id'] for node in graph_data['nodes']]
node_mapping = {node_id: i for i, node_id in enumerate(node_list)}
node_index = {index: node for node, index in node_mapping.items()}

# Create a list of edges in PyTorch Geometric format
edge_index = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
features = torch.randn(len(node_list), 1)
labels = torch.tensor(list(range(len(graph_data['nodes']))), dtype=torch.long)

large_dataset = Data(x=features, edge_index=edge_index, y=labels, node_mapping=node_mapping, node_index=node_index)
torch.save(large_dataset, 'large_dataset.pth')
large_dataset.cuda()
```

Next, it is necessary to generate subgraphs for training the model. This can be done using the following code:

```python
import json
import os
from redkg.generate_subgraphs import generate_subgraphs

# Generate a dataset of 1000 subgraphs, each containing between 3 and 15 nodes
if not os.path.isfile('subgraphs.json'):
    subgraphs = generate_subgraphs(graph_data, num_subgraphs=1000, min_nodes=3, max_nodes=15)
    with open('subgraphs.json', 'w') as f:
        json.dump(subgraphs, f)
else:
    with open('subgraphs.json', 'r') as f:
        subgraphs = json.load(f)
```

Next, convert the subgraphs into PyTorch Geometric format:

```python
from redkg.generate_subgraphs import generate_subgraphs_dataset

dataset = generate_subgraphs_dataset(subgraphs, large_dataset)
```

### Model Training

Let's initialize the optimizer and the model in training mode:

```python
from redkg.models.graphsage import GraphSAGE
from torch.optim import Adam

# Train the GraphSAGE model (GCN or GAT can also be used)
#   number of input and output features matches the number of nodes in the large graph - 177
#   number of layers - 64
model = GraphSAGE(large_dataset.num_node_features, 64, large_dataset.num_node_features)
model.train()

# Use the Adam optimizer
#   learning rate - 0.0001
#   weight decay - 1e-5
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
```

Start training the model for 2 epochs:

```python
from redkg.train import train_gnn_model
from redkg.negative_samples import generate_negative_samples

# Model training
loss_values = []
for epoch in range(2):
    for subgraph in dataset:
        positive_edges = subgraph.edge_index.t().tolist()
        negative_edges = generate_negative_samples(subgraph.edge_index, subgraph.num_nodes, len(positive_edges))
        if len(negative_edges) == 0:
            continue
        loss = train_gnn_model(model, optimizer, subgraph, positive_edges, negative_edges)
        loss_values.append(loss)
        print(f"Epoch: {epoch}, Loss: {loss}")
```

## Architecture Overview

ReDKG is a framework implementing strong AI algorithms for deep learning with reinforcement on dynamic knowledge graphs for decision support tasks. The figure below shows the general structure of the component. It includes four main modules:

* Graph encoding modules into vector representations (encoder):
  * KGE, implemented using the KGEModel class in `redkg.models.kge`
  * GCN, implemented using the GCN class in `redkg.models.gcn`
  * GAT, implemented using the GAT class in `redkg.models.gat`
  * GraphSAGE, implemented using the GraphSAGE class in `redkg.models.graphsage`
* State representation module (state representation), implemented using the GCNGRU class in `redkg.models.gcn_gru_layers`
* Candidate object selection module (action selection)

Project Structure
=================

The latest stable release of ReDKG is in the [`main branch`](https://github.com/aimclub/ReDKG)

The repository includes the following directories:
* Package `redkg` contains the main classes and scripts;
* Package `examples` includes several *how-to-use-cases* where you can start to discover how ReDKG works;
* Directory `data` shoul be contains data for modeling;
* All *unit and integration tests* can be observed in the `test` directory;
* The sources of the documentation are in the `docs`.


Cases and examples
==================
To learn representations with default values of arguments from command line, use:
```
python kg_run
```

To learn representations in your own project, use:

```python
from kge import KGEModel
from edge_predict import Evaluator
evaluator = Evaluator()

kge_model = KGEModel(
        model_name="TransE",
        nentity=info['nentity'],
        nrelation=info['nrelation'],
        hidden_dim=128,
        gamma=12.0,
        double_entity_embedding=True,
        double_relation_embedding=True,
        evaluator=evaluator
    )
```

### Train KGQR model
To train KGQR model on your own data:
```
negative_sample_size = 128
nentity = len(entity_vocab.keys())
train_count = calc_state_kg(triples)

dataset = TrainDavaset (triples,
                        nentity,
                        len(relation_vocab.keys()),
                        negative_sample_size,
                        "mode",
                        train_count)

conf = Config()

#Building Net
model = GCNGRU(Config(), entity_vocab, relation_vocab, 50)

# Embedding pretrain by TransE
crain_kge_model (model_kge_model, train pars, info, triples, None)

#Training using RL
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(Config(), item_vocab, model, optimizer)

```

### Visualization of graphs and hypergraphs
The [`reference books`](./redkg/visualization/config/parameters/) system is used to visualize graphs and hypergraphs to set visualization parameters, as well as the [`contracts`](./redkg/visualization/contracts/) system to install graphic elements.

Graph visualization example:
```python
graph_contract: GraphContract = GraphContract(
    vertex_num=10,
    edge_list=(
        [(0, 7), (2, 7), (4, 9), (3, 7), (1, 8), (5, 7), (2, 3), (4, 5), (5, 6), (4, 8), (6, 9), (4, 7)],
        [1.0] * 12,
    ),
    edge_num=12,
    edge_weights=list(tensor([1.0] * 24)),
)

vis_contract: GraphVisualizationContract = GraphVisualizationContract(graph=graph_contract)

vis: GraphVisualizer = GraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()
```

Hypergraph visualization example:
```python
graph_contract: HypergraphContract = HypergraphContract(
    vertex_num=10,
    edge_list=(
        [(3, 4, 5, 9), (0, 4, 7), (4, 6), (0, 1, 2, 4), (3, 6), (0, 3, 9), (2, 5), (4, 7)],
        [1.0] * 8,
    ),
    edge_num=8,
    edge_weights=list(tensor([1.0] * 10)),
)

vis_contract: HypergraphVisualizationContract = HypergraphVisualizationContract(graph=graph_contract)

vis: HypergraphVisualizer = HypergraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()
```

### Visualization of graphs and hypergraphs
The [`reference books`](./redkg/visualization/config/parameters/) system is used to visualize graphs and hypergraphs to set visualization parameters, as well as the [`contracts`](./redkg/visualization/contracts/) system to install graphic elements.

Graph visualization example:
```python
graph_contract: GraphContract = GraphContract(
    vertex_num=10,
    edge_list=(
        [(0, 7), (2, 7), (4, 9), (3, 7), (1, 8), (5, 7), (2, 3), (4, 5), (5, 6), (4, 8), (6, 9), (4, 7)],
        [1.0] * 12,
    ),
    edge_num=12,
    edge_weights=list(tensor([1.0] * 24)),
)

vis_contract: GraphVisualizationContract = GraphVisualizationContract(graph=graph_contract)

vis: GraphVisualizer = GraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()
```

Hypergraph visualization example:
```python
graph_contract: HypergraphContract = HypergraphContract(
    vertex_num=10,
    edge_list=(
        [(3, 4, 5, 9), (0, 4, 7), (4, 6), (0, 1, 2, 4), (3, 6), (0, 3, 9), (2, 5), (4, 7)],
        [1.0] * 8,
    ),
    edge_num=8,
    edge_weights=list(tensor([1.0] * 10)),
)

vis_contract: HypergraphVisualizationContract = HypergraphVisualizationContract(graph=graph_contract)

vis: HypergraphVisualizer = HypergraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()
```

### BellmanFordLayerModified

`BellmanFordLayerModified` is a PyTorch layer implementing a modified Bellman-Ford algorithm for analyzing graph properties and extracting features from graph structures. This layer can be used in graph machine learning tasks such as path prediction and graph structure analysis.

### Usage

```python
import torch
from raw_bellman_ford.layers.bellman_ford_modified import BellmanFordLayerModified

# Initialize the layer with the number of nodes and the number of features
num_nodes = 4
num_features = 5
bellman_ford_layer = BellmanFordLayerModified(num_nodes, num_features)

# Define the adjacency matrix of the graph and the source node
adj_matrix = torch.tensor([[0, 2, float('inf'), 1],
                          [float('inf'), 0, -1, float('inf')],
                          [float('inf'), float('inf'), 0, -2],
                          [float('inf'), float('inf'), float('inf'), 0]])
source_node = 0

# Calculate graph features, diameter, and eccentricity
node_features, diameter, eccentricity = bellman_ford_layer(adj_matrix, source_node)

print("Node Features:")
print(node_features)
print("Graph Diameter:", diameter)
print("Graph Eccentricity:", eccentricity)
```

### Layer Parameters

- `num_nodes`: The number of nodes in the graph.
- `num_features`: The number of features extracted from the graph.
- `edge_weights`: Weights of edges between nodes (trainable parameters).
- `node_embedding`: Node embedding for feature extraction.

### Application

- `BellmanFordLayerModified` is useful when additional graph characteristics, such as diameter and eccentricity, are of interest along with paths.

### HypergraphCoverageSolver

`HypergraphCoverageSolver` is a Python class representing an algorithm to solve the coverage problem for a hypergraph. The problem is to determine whether an Unmanned Aerial Vehicle (UAV) can cover all objects in the hypergraph, taking into account the UAV's radius of action.

### Usage

```python
from raw_bellman_ford.algorithms.coverage_solver import HypergraphCoverageSolver

# Define nodes, edges, hyperedges, node types, edge types, and hyperedge types
nodes = [1, 2, 3, 4, 5]
edges = [(1, 2), (2, 3), (3, 1), ((1, 2, 3), 4), ((1, 2, 3), 5), (4, 5)]
hyperedges = [(1, 2, 3)]
node_types = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
edge_types = {(1, 2): 1.4, (2, 3): 1.5, (3, 1): 1.6, ((1, 2, 3), 4): 2.5, ((1, 2, 3), 5): 24.6, (4, 5): 25.7}
hyperedge_types = {(1, 2, 3): 1}

# Create an instance of the class
hypergraph_solver = HypergraphCoverageSolver(nodes, edges, hyperedges, node_types, edge_types, hyperedge_types)
```

### Define UAV Radius and Check Coverage Feasibility

```python
# Define UAV radius
drone_radius = 40

# Check if the UAV can cover all objects in the hypergraph
if hypergraph_solver.can_cover_objects(drone_radius):
    print("The UAV can cover all objects in the hypergraph.")
else:
    print("The UAV cannot cover all objects in the hypergraph.")
```

### How It Works

The algorithm for solving the hypergraph coverage problem first calculates the minimum radius required to cover all objects. It considers both regular edges and hyperedges, taking into account their weights. Then, the algorithm compares the computed minimum radius with the UAV's radius of action. If the UAV's radius is not less than the minimum radius, it is considered that the UAV can cover all objects in the hypergraph.

Documentation
=============
Detailed information and description of ReDKG framework is available in the [`Documentation`](https://aimclub.github.io/ReDKG/)

## Contribution
To contribute this library, the current [code and documentation convention](/wiki/Development.md) should be followed.
Project run linters and tests on each pull request, to install linters and testing-packages locally, run 

```
pip install -r requirements-dev.txt
```
To avoid any unnecessary commits please fix any linting and testing errors after running of the each linter:
- `pflake8 .`
- `black .`
- `isort .`
- `mypy stable_gnn`
- `pytest tests`

Contacts
========
- [Contact development team](mailto:egorshikov@itmo.ru)
- Natural System Simulation Team <https://itmo-nss-team.github.io/>

## Suported by
The study is supported by the [Research Center Strong Artificial Intelligence in Industry](https://sai.itmo.ru/) of [ITMO University](https://itmo.ru/) as part of the plan of the center's program: Development and testing of an experimental sample of the library of algorithms of strong AI in terms of deep reinforcement learning on dynamic knowledge graphs for decision support tasks

Citation
========
```
@article{EGOROVA2022284,
title = {Customer transactional behaviour analysis through embedding interpretation},
author = {Elena Egorova and Gleb Glukhov and Egor Shikov},
journal = {Procedia Computer Science},
volume = {212},
pages = {284-294},
year = {2022},
doi = {https://doi.org/10.1016/j.procs.2022.11.012},
url = {https://www.sciencedirect.com/science/article/pii/S1877050922017033}
}
```
