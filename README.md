# Auspex

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

It is required to install [Anaconda environment](https://docs.anaconda.com/anaconda/install/).

Once anaconda is installed, create a new virtual environment using the provided environment.yml

```
conda env create -f environment.yml

```
Activate the new environment with
```
conda activate auspex
```


### Installing


Install Auspex through **pip install** from its Git repository.


```
pip install git+https://github.com/thorn-lab/auspex_python.git
```

### Usage

To use Auspex, use following console command:

```
auspex [path_to_data_file] [options]
```

To read the detailed help message:

```
auspex -h
```


### Tests

```
auspex src/test/4puc.mtz
```

### Documentation
You can generate the documentation with [sphinx](https://www.sphinx-doc.org/en/master/index.html)
```
cd docs
make html
your_favorite_browser build/auspex.html
```
