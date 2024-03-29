# BellmanFordLayerModified

`BellmanFordLayer` - это PyTorch-слой, предоставляющий результаты алгоритма Беллмана-Форда для анализа графовых данных. Он возвращает матрицу расстояний и матрицу предшественников, которые могут быть использованы для поиска кратчайших путей в графе. Также этот слой определяет наличие отрицательных циклов в графе.

`   ` - это PyTorch слой, реализующий модифицированный алгоритм Беллмана-Форда для анализа свойств графов и извлечения признаков из графовой структуры. Этот слой может использоваться в задачах графового машинного обучения, таких как предсказание путей и анализ графовых структур.

## Использование

```python
import torch

# Инициализация слоя с указанием количества узлов и числа признаков
num_nodes = 4
num_features = 5
bellman_ford_layer = BellmanFordLayerModified(num_nodes, num_features)

# Определение матрицы смежности графа и начального узла
adj_matrix = torch.tensor([[0, 2, float('inf'), 1],
                          [float('inf'), 0, -1, float('inf')],
                          [float('inf'), float('inf'), 0, -2],
                          [float('inf'), float('inf'), float('inf'), 0]])
source_node = 0

# Вычисление признаков графа, диаметра и эксцентриситета
node_features, diameter, eccentricity = bellman_ford_layer(adj_matrix, source_node)

print("Node Features:")
print(node_features)
print("Graph Diameter:", diameter)
print("Graph Eccentricity:", eccentricity)
```

## Параметры слоя

- `num_nodes`: Количество узлов в графе.
- `num_features`: Количество признаков, извлекаемых из графа.
- `edge_weights`: Веса ребер между узлами (обучаемые параметры).
- `node_embedding`: Вложение узлов для извлечения признаков.

## Применение:

- BellmanFordLayer полезен, когда вам нужны результаты алгоритма Беллмана-Форда для выполнения других операций или анализа графа.
- BellmanFordLayerModified полезен, когда вас помимо путей интересуют дополнительные характеристики графа, такие как диаметр и эксцентриситет.

## Алгоритм Bellman-Ford для Гиперграфа

Эта Python-реализация предоставляет версию алгоритма Bellman-Ford для гиперграфа. Алгоритм предназначен для вычисления расстояний между узлами, а также для определения эксцентриситета, радиуса и диаметра гиперграфа. Кроме того, предоставлены методы для вычисления центральности узлов.

## Использование

### 1. Импорт класса и создание экземпляра

```python
# Задайте узлы, ребра, гиперребра, типы узлов, типы ребер, веса ребер, типы гиперребер и веса гиперребер
nodes = [1, 2, 3, 4, 5]
edges = [(1, 2), (2, 3), (2, 1), (3, 4), (3, 5)]
hyperedges = [(1, 2, 3), (3, 5, 4)]
node_types = {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
edge_types = {(1, 2): 1, (2, 3): 2, (2, 1): 1, (3, 4): 1, (3, 5): 1}
edge_weights = {(1, 2): 5.4, (2, 3): 2.2, (2, 1): 2.1, (3, 4): 1, (3, 5): 1}
hyperedge_types = {(1, 2, 3): 1, (3, 5, 4): 1}
hyperedge_weights = {(1, 2, 3): 3, (3, 5, 4): 2}

# Создайте экземпляр класса
bellman_ford = HBellmanFord(nodes, edges, node_types, edge_types, edge_weights, hyperedges, hyperedge_types, hyperedge_weights)
```

### 2. Определение критериев

```python
# Критерии для узлов, ребер и гиперребер на основе типов. 0 обозначает учет всех типов.
node_criteria_all_types = 0
node_criteria_specific_types = [0, 1]
edge_criteria_all_types = 0
edge_criteria_specific_types = [1]
hyperedge_criteria_vertices = 0
hyperedge_criteria_hyperedges = [1]
```

### 3. Запуск алгоритма Bellman-Ford

```python
# Запустите алгоритм Bellman-Ford
distance_matrix = bellman_ford.bellman_ford(node_criteria_all_types, edge_criteria_all_types, hyperedge_criteria_vertices)
```
### 5. Вывод результатов

```python
# Выведите результаты
distances = np.array(distance_matrix)
print("Eccentricity:", bellman_ford.eccentricity(distances))
print("Radius:", bellman_ford.radius(distances))
print("Diameter:", bellman_ford.diameter(distances))
print("Central Nodes:", bellman_ford.central_nodes(distances))
print("Peripheral Nodes:", bellman_ford.peripheral_nodes(distances))

for node in range(len(nodes)):
    print(f"\nЦентральность узла {node}:")
    print("Центральность близости:", bellman_ford.closeness_centrality(node, distances))
    print("Центральность степени:", bellman_ford.degree_centrality(node))
    print("Центральность посредничества:", bellman_ford.betweenness_centrality(node, distances))
```

## Метрики

- **Eccentricity (Эксцентриситет):** Максимальное расстояние от узла до всех остальных узлов.
- **Radius (Радиус):** Минимальное эксцентриситета по всем узлам.
- **Diameter (Диаметр):** Максимальное расстояние между парами узлов.
- **Central Nodes (Центральные узлы):** Узлы с минимальным эксцентриситетом, образующие центр графа.
- **Peripheral Nodes (Периферийные узлы):** Узлы с максимальным эксцентриситетом, находящиеся на окраине графа.
- **Closeness Centrality (Центральность близости):

** Обратная сумма расстояний от узла до всех остальных узлов. Изолированным узлам присваивается значение 0.
- **Degree Centrality (Центральность степени):** Доля узлов, с которыми связан данный узел, в зависимости от общего количества узлов в графе.
- **Betweenness Centrality (Центральность посредничества):** Сумма долей кратчайших путей, проходящих через узел, относительно всех кратчайших путей в графе.

## Алгоритм решения задачи покрытия для Гиперграфа

Эта Python-реализация представляет алгоритм для решения задачи о покрытии для гиперграфа. Задача заключается в определении, может ли Беспилотный Летательный Аппарат (БПЛА) покрыть все объекты в гиперграфе, учитывая радиус действия БПЛА.

## Использование

### 1. Импорт класса и создание экземпляра

```python
# Задайте узлы, ребра, гиперребра, типы узлов, типы ребер и типы гиперребер
nodes = [1, 2, 3, 4, 5]
edges = [(1, 2), (2, 3), (3, 1), ((1, 2, 3), 4), ((1, 2, 3), 5), (4, 5)]
hyperedges = [(1, 2, 3)]
node_types = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
edge_types = {(1, 2): 1.4, (2, 3): 1.5, (3, 1): 1.6, ((1, 2, 3), 4): 2.5, ((1, 2, 3), 5): 24.6, (4, 5): 25.7}
hyperedge_types = {(1, 2, 3): 1}

# Создайте экземпляр класса
hypergraph_solver = HypergraphCoverageSolver(nodes, edges, hyperedges, node_types, edge_types, hyperedge_types)
```

### 2. Определение радиуса БПЛА

```python
# Задайте радиус действия БПЛА
drone_radius = 40
```

### 3. Проверка возможности покрытия объектов

```python
# Проверьте, может ли БПЛА покрыть все объекты в гиперграфе
if hypergraph_solver.can_cover_objects(drone_radius):
    print("БПЛА может покрыть все объекты в гиперграфе.")
else:
    print("БПЛА не может покрыть все объекты в гиперграфе.")
```

## Как это работает

Алгоритм решения задачи покрытия для гиперграфа сначала вычисляет минимальный радиус, необходимый для покрытия всех объектов. Для этого он рассматривает как обычные ребра, так и гиперребра, учитывая их веса. Затем алгоритм сравнивает вычисленный минимальный радиус с радиусом действия БПЛА. Если радиус БПЛА не меньше минимального радиуса, то считается, что БПЛА может покрыть все объекты в гиперграфе.