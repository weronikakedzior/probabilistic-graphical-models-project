# Probabilistic Graphical Models - project 1

Preparation of the environment consists in installing the required packages from the supplied file `requirements.txt` (the project was tested in `Python 3.8.2` environment):
```console
pip install -r requirements.txt
```

To perform the experiments, run the `run.py` script in the `src/` folder. 
You also need to specify the parameter that will be predicted ('class' od 'n_children'). Usage:

```console
cd src
python run.py class
```
or
```console
cd src
python run.py n_children
```

The results will be saved to the `out` folder in `.csv` format.

Authors:
- Marcel Cielinski
- Weronika Kędzior (Pawlak)
