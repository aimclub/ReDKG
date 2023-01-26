# ReDKG
[![Documentation](https://github.com/aimclub/ReDKG/actions/workflows/gh_pages.yml/badge.svg)](https://aimclub.github.io/ReDKG/)
[![Linters](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/linters.yml)
[![Tests](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml/badge.svg)](https://github.com/aimclub/ReDKG/actions/workflows/tests.yml)
<p align="center">
  <img src="https://github.com/aimclub/ReDKG/blob/main/docs/img/logo.png?raw=true" width="300px"> 
</p>


**Re**inforcement learning on **D**ynamic **K**nowledge **G**raphs (**ReDKG**) --
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

---------
<p align="center">
  <img src="https://github.com/aimclub/ReDKG/blob/main/docs/img/lib_schema.png.png?raw=true" width="800px"> 
</p>
Более подробную информацию можно найти в [кратком руководстве](qwe.asd), а также в [учебном пособии](qwe.asd). 

Структура проекта
=================

Последняя стабильная версия проекта проекта доступна по [ссылке](https://github.com/aimclub/ReDKG)

Репозиторий проекта включает в себя следующие директории:
* директория `redkg` - содержит основные классы и функции проекта;
* директория `examples` - содержит несколько примеров использования;
* директория `data` - должна содержать данные для моделирования;
* все *модульные и интеграционные тесты* можно посмотреть в директории `test`;
* исходная документация содержится в директории `docs`.


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

Документация
=============
Подробная информация и описание библиотеки ReDKG доступны по [`ссылке`](https://aimclub.github.io/ReDKG/)

## Внесение своего вклада в проект
Для внесения своего вклада в проект необходимо следовать текущему [соглашению о коде и документации](wiki/Development.md).
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
Исследование выполнено при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru/) (Санкт-Петербург, Россия)

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
