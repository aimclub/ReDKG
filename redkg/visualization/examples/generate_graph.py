"""Generate graph module."""

from redkg.visualization.data_generation.graph_generator import GraphGenerator

generator = GraphGenerator(vertex_num=4, edge_num=5)
data = generator()
print(data)
