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

#### PyTorch 1.12

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

После установки необходимых библиотек необходимо скопировать репозиторий и выполнить следующую команду, находясь в директории проекта:

```
pip install . 
```

## Загрузка тестовых данных
Тестовый набор данных может быть получен по ссылке [movielens](https://grouplens.org/datasets/movielens/20m/) и распакован в директорию /data/.
После распаковки директория /data/ должна содержать следующий набор файлов: 
 - `ratings.csv` - исходный файл оценок пользователей;
 - `attributes.csv` - исходный файл атрибутов объектов;
 - `kg.txt` - файл, содержащий граф знаний;
 - `item_index2enity_id.txt` - сопоставление индексов объектов в исходном файле оценок пользователей с индексами объектов в графе знаний;
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



Обзор Архитектуры 
=================

ReDKG - это фреймворк, реализующий алгоритмы сильного ИИ в части глубокого обучения с подкреплением на динамических графах знаний для задач поддержки принятия решений. На рисунке ниже приведена общая структура компонента. Она включает в себя четыре основных модуля: 
 * модуль кодирования графа в векторные представления (кодировщик), реализованный с помощью класса KGEModel в файлу redkg.models;
 * модуль представления состояния (представление состояния), реализованный с помощью класса GCNGRU в файлу redkg.models;
 * модуль выбора объектов-кандидатов (отбор возможных действий);
 * модуль Q-обучения (Q-network)  , реализованный классом TrainPipeline в файле redkg.train. 

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
* директория `raw_bellman_ford` - содержит модульную реализацию алгоритма Беллмана-Форда;
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

### BellmanFordLayerModified

`BellmanFordLayerModified` - это PyTorch-слой, реализующий модифицированный алгоритм Беллмана-Форда для анализа свойств графов и извлечения признаков из графовой структуры. Этот слой может использоваться в задачах графового машинного обучения, таких как предсказание путей и анализ графовых структур.

### Использование

```python
import torch
from raw_bellman_ford.layers.bellman_ford_modified import BellmanFordLayerModified

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

### Параметры слоя

- `num_nodes`: Количество узлов в графе.
- `num_features`: Количество признаков, извлекаемых из графа.
- `edge_weights`: Веса ребер между узлами (обучаемые параметры).
- `node_embedding`: Вложение узлов для извлечения признаков.

### Применение

- `BellmanFordLayerModified` полезен, когда вас помимо путей интересуют дополнительные характеристики графа, такие как диаметр и эксцентриситет.

### HypergraphCoverageSolver

`HypergraphCoverageSolver` - это Python-класс, представляющий алгоритм решения задачи покрытия для гиперграфа. Задача заключается в определении, может ли Беспилотный Летательный Аппарат (БПЛА) покрыть все объекты в гиперграфе, учитывая радиус действия БПЛА.

### Использование

```python
from raw_bellman_ford.algorithms.coverage_solver import HypergraphCoverageSolver

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

### Определение радиуса БПЛА и проверка возможности покрытия объектов

```python
# Задайте радиус действия БПЛА
drone_radius = 40

# Проверьте, может ли БПЛА покрыть все объекты в гиперграфе
if hypergraph_solver.can_cover_objects(drone_radius):
    print("БПЛА может покрыть все объекты в гиперграфе.")
else:
    print("БПЛА не может покрыть все объекты в гиперграфе.")
```

### Как это работает

Алгоритм решения задачи покрытия для гиперграфа сначала вычисляет минимальный радиус, необходимый для покрытия всех объектов. Для этого он рассматривает как обычные ребра, так и гиперребра, учитывая их веса. Затем алгоритм сравнивает вычисленный минимальный радиус с радиусом действия БПЛА. Если радиус БПЛА не меньше минимального радиуса, то считается, что БПЛА может покрыть все объекты в гиперграфе.


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

Документация
=============
Подробная информация и описание библиотеки ReDKG доступны по [`ссылке`](https://aimclub.github.io/ReDKG/)

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
