**Re**inforcement learning on **D**ynamic **K**nowledge **G**raphs (**ReDKG**) is a toolkit for deep reinforcement learning on dynamci knowledge graphs.
==========
**Re**inforcement learning on **D**ynamic **K**nowledge **G**raphs (**ReDKG**) is a toolkit for deep reinforcement learning on dynamci knowledge graphs. it is designed to encode static and dynamic (temporary) knowledge graphs (KG) by constructing vector representations for the entities and relationships included in them.  The algorithm implements the functions of strong AI in terms of supporting algorithms for strong AI. The algorithm of coding of knowledge graphs is used to obtain vector representations of KG and DKG, which are used to build a DSS using algorithms of strong AI learning with reinforcement on vector representations of KG and DKG.

![plot](/docs/img/lib_schema.png)

How to use
==========
 1. Clone repository
```
git clone ...
```
2. Install requirements
```python
pip install -r /path/to/requirements.txt
```
3. Donwload test data
Download [ratings.csv](https://grouplens.org/datasets/movielens/20m/) to /data/ folder./
Data folder should be contains three files: 
 - ratings.csv - raw rating file of Movielens-20M dataset;
 -   kg.txt - knowledge graph file;
 -  item_index2enity_id.txt - the mapping from item indices in the raw rating file to entity IDs in the KG file;
4. Run preprocessing.py
```python
python preprocess.py
```
5. Run train.py
```python train.py
python preprocess.py
```
\
More details about first steps with  might be found in the [quick start guide](qwe.asd) and in the [tutorial for novices](qwe.asd).



 Dependencies
=========
 - Python >= 3.7.0 
 - numpy >= 1.15.4
 - pandas >= 1.0.0
 - PyTorch >= 1.0.0
 - PyTorch Geometric >= 1.6.0
 - NetworkX >= 2.5.0

Project Structure
=================

The latest stable release of  is on the

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

To learn representations in ypur own project, use:

```python
from kge import KGEModel
from edge_predict import Evaluator
evaluator = Evaluator()
kge_model = KGEModel(model_name = MODEL_NAME, nentity=nentity, nrelation=nrelation, embedding_size=EMBEDDING_SIZE, gamma=GAMMA, evaluator=evaluator)
```

### Train KGQR model
To train KGQR model on your own data ...
```
...
```

Documentation
=============
Detailed information and description of ReDKG framework is available in the [`Documentation`](link)

Contacts
========
- [Contact development team](mailto:egorshikov@itmo.ru)
- Natural System Simulation Team <https://itmo-nss-team.github.io/>

Supported by
============

[National Center for Cognitive Research of ITMO University](https://actcognitive.org/) 

Citation
========
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

<!---
.. |docs| image:: https://readthedocs.org/projects/gefest/badge/?version=latest
   :target: https://gefest.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |license| image:: https://img.shields.io/github/license/ITMO-NSS-team/GEFEST
   :alt: Supported Python Versions
   :target: ./LICENSE.md

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :target: https://t.me/gefest_helpdesk
   :alt: Telegram Chat
--->
