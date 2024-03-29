# AUSPEX

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The easiest approach to install AUSPEX is through conda environment.

Either install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

Once installed, create a new virtual environment using the provided environment.yml

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
pip install git+https://github.com/thorn-lab/AUSPEX.git
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
