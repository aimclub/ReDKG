# ReDKG
[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)
[![Documentation](https://github.com/aimclub/ReDKG/actions/workflows/gh_pages.yml/badge.svg)](https://aimclub.github.io/ReDKG/)
[![Linters](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml)
[![Tests](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml)
[![Mirror](https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162)](https://gitlab.actcognitive.org/itmo-sai-code/ReDKG)
<p align="center">
  <img src="https://github.com/aimclub/ReDKG/blob/main/docs/img/logo.png?raw=true" width="300px"> 
</p>


**Re**inforcement learning on **D**ynamic **K**nowledge **G**raphs (**ReDKG**) -
это библиотека глубокого обучения с подкреплением на динамических графах знаний. 

Библиотека предназначена для кодирования статических и динамических графов знаний (ГЗ) при помощи построения векторных представлений сущностей и отношений.
Алгоритм обучения с подкреплением на основе векторных представлений предназначен для обучения рекомендательных моделей и моделей систем поддержки принятия решений на основе обучения с подкреплением (RL) на основе векторных представлений графов.


## Установка

Для работы библиотеки необходим интерпретатор языка программирования Python версии не ниже 3.9 

На первом шаге необходимо выполнить установку библиотеки, [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric/) и Torch 1.1.2.

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

После установки необходимых библиотек необходимо скопировать репозиторий и выполнить следующую команду, находясь в
директории проекта:

```
pip install . 
```

## Пример использования модели KGE

### Загрузка тестовых данных

Тестовый набор данных может быть получен по ссылке [movielens](https://grouplens.org/datasets/movielens/20m/) и
распакован в директорию `/data/`.

После распаковки директория /data/ должна содержать следующий набор файлов:

- `ratings.csv` - исходный файл оценок пользователей;
- `attributes.csv` - исходный файл атрибутов объектов;
- `kg.txt` - файл, содержащий граф знаний;
- `item_index2enity_id.txt` - сопоставление индексов объектов в исходном файле оценок пользователей с индексами объектов
  в графе знаний;

### Предобработка данных

```python
from redkg.config import Config
from redkg.preprocess import DataPreprocessor

config = Config()
preprocessor = DataPreprocessor(config)
preprocessor.process_data()
```

### Обучение модели

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

## Пример использования моделей GCN, GAT, GraphSAGE

Данные модели реализуют алгоритм предсказания связей в графе знаний.

Дополнительную информацию о шагах обучения можно найти в примере [basic_link_prediction.ipynb](https://github.com/aimclub/ReDKG/blob/main/examples/basic_link_prediction.ipynb).

### Загрузка тестовых данных

Тестовый набор данных может быть получен по ссылке [jd_data2.json](https://github.com/ZhongTr0n/JD_Analysis/blob/main/jd_data2.json)
и положен в директорию `/data/`.

### Предобработка данных

Для предобработки потребуется выполнить чтение данных из файла и преобразование их в формат PyTorch Geometric.

```python
import json
import torch
from torch_geometric.data import Data

# Прочтём данные из файла
with open('jd_data2.json', 'r') as f:
    graph_data = json.load(f)

# Извлечём список узлов и преобразуем его в словарь для быстрого поиска
node_list = [node['id'] for node in graph_data['nodes']]
node_mapping = {node_id: i for i, node_id in enumerate(node_list)}
node_index = {index: node for node, index in node_mapping.items()}

# Создадим список рёбер в формате PyTorch Geometric
edge_index = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
features = torch.randn(len(node_list), 1)
labels = torch.tensor(list(range(len(graph_data['nodes']))), dtype=torch.long)

large_dataset = Data(x=features, edge_index=edge_index, y=labels, node_mapping=node_mapping, node_index=node_index)
torch.save(large_dataset, 'large_dataset.pth')
large_dataset.cuda()
```

Далее необходимо сгенерировать подграфы для обучения модели. Для этого можно использовать следующий код:

```python
import json
import os
from redkg.generate_subgraphs import generate_subgraphs

# Сгенерируем датасет на 1000 подграфов, каждый из которых будет содержать от 3 до 15 узлов
if not os.path.isfile('subgraphs.json'):
    subgraphs = generate_subgraphs(graph_data, num_subgraphs=1000, min_nodes=3, max_nodes=15)
    with open('subgraphs.json', 'w') as f:
        json.dump(subgraphs, f)
else:
    with open('subgraphs.json', 'r') as f:
        subgraphs = json.load(f)
```

Далее необходимо выполнить преобразование подграфов в формат PyTorch Geometric:

```python
from redkg.generate_subgraphs import generate_subgraphs_dataset

dataset = generate_subgraphs_dataset(subgraphs, large_dataset)
```

### Обучение модели

Выполним инициализацию оптимизатора и модели в режиме обучения:

```python
from redkg.models.graphsage import GraphSAGE
from torch.optim import Adam

# Обучим модель GraphSAGE (так же можно использовать GCN или GAT)
#   количество входных и выходных признаков совпадает с количеством узлов в большом графе - 177
#   количество слоёв - 64
model = GraphSAGE(large_dataset.num_node_features, 64, large_dataset.num_node_features)
model.train()

# Используем оптимизатор Adam
#   скорость обучения - 0.0001
#   весовой коэффициент - 1e-5
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
```

Запустим обучение модели в 2 эпохи:

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

## Обзор Архитектуры 

ReDKG - это фреймворк, реализующий алгоритмы сильного ИИ в части глубокого обучения с подкреплением на динамических
графах знаний для задач поддержки принятия решений. На рисунке ниже приведена общая структура компонента. Она включает в
себя четыре основных модуля:

* модули кодирования графа в векторные представления (кодировщик):
  * KGE, реализованный с помощью класса KGEModel в `redkg.models.kge`
  * GCN, реализованный с помощью класса GCN в `redkg.models.gcn`
  * GAT, реализованный с помощью класса GAT в `redkg.models.gat`
  * GraphSAGE, реализованный с помощью класса GraphSAGE в `redkg.models.graphsage`
* модуль представления состояния (представление состояния), реализованный с помощью класса GCNGRU в `redkg.models.gcn_gru_layers`
* модуль выбора объектов-кандидатов (отбор возможных действий)
* модуль Q-обучения (Q-network), реализованный классом TrainPipeline в `redkg.train`

<p align="center">
  <img src="https://github.com/aimclub/ReDKG/blob/main/docs/img/lib_schema_ru.png?raw=true" width="700px"> 
</p>

Структура проекта
=================

Последняя стабильная версия проекта проекта доступна по [ссылке](https://github.com/aimclub/ReDKG)

Репозиторий проекта включает в себя следующие директории:
* директория `redkg` - содержит основные классы и функции проекта;
* директория `examples` - содержит несколько примеров использования;
* директория `data` - должна содержать данные для моделирования;
* все *модульные и интеграционные тесты* можно посмотреть в директории `test`;
* документация содержится в директории `docs`.


Примеры использования
==================


Для обучения векторных представлений с параметрами по умолчанию выполните следующую команду в командной строке:
```
python kg_run
```

Для обучение векторных представлений в своем проекте используйте:

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

### Обучение модели KGQR
Для обучения модели KGQR на собственных данных используйте:
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

### Визуализация графов и гиперграфов
Для визуализации графов и гиперграфов используется система [`справочников`](./redkg/visualization/config/parameters/) для установки параметров визуализации, а также система [`контрактов`](./redkg/visualization/contracts/) для установки графических элементов.

Пример визуализации графа:
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

Пример визуализации гиперграфа:
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

Документация
=============
Подробная информация и описание библиотеки ReDKG доступны по [`ссылке`](https://aimclub.github.io/ReDKG/)

## Внесение своего вклада в проект
Для внесения своего вклада в проект необходимо следовать текущему [соглашению о коде и документации](/wiki/Development.md).
Проект запускает линтеры и тесты для каждого реквест-запроса, чтобы установить линтеры и тестовые пакеты локально, запустите

```
pip install -r requirements-dev.txt
```
Для избежания лишних коммитов, пожалуйста, исправьте все ошибки после запуска каждого линтера:
- `pflake8 .`
- `black .`
- `isort .`
- `mypy stable_gnn`
- `pytest tests`

Контакты
========
- [Разработчик](mailto:egorshikov@itmo.ru)
- Natural System Simulation Team <https://itmo-nss-team.github.io/>

## Поддержка
Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru/) в рамках мероприятия программы центра: Разработка и испытания экспериментального образца библиотеки алгоритмов сильного ИИ в части глубокого обучения с подкреплением на динамических графах знаний для задач поддержки принятия решений

<p align="center">
  <img src="https://github.com/anpolol/StableGNN/blob/main/docs/AIM-logo.svg?raw=true" width="300px"> 
</p>


## Цитирование
Если используете библиотеку в ваших работах, пожалуйста, процитируйте [статью](https://www.sciencedirect.com/science/article/pii/S1877050922017033) (и другие соответствующие статьи используемых методов):

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
