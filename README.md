# AUSPEX

AUSPEX is a diagnostic tool for graphical X-Ray data analysis, it enables users to 
automatically detect 
**ice-ring artefacts with HELCARAXE CNN network**, 
**beamstop outliers with NEMOs culstering strategy and semi-supervised learning**, 
or visualize other problems in integrated X-ray diffraction data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites (without Docker)

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
auspex test/4puc.mtz
auspex test/8g0s.mtz --beamstop_outlier
auspex test/5usx.mtz --nemo-removal --generate-xds-filter test/5usx_INTEGRATE.HKL
```

### Use Docker Container

To run Auspex with docker

```
docker build --tag auspex .
```

Test with:

```
docker run -v test:/app auspex /app/4puc.mtz
```

### Documentation
You can generate the documentation with [sphinx](https://www.sphinx-doc.org/en/master/index.html)
```
cd docs
make html
your_favorite_browser build/auspex.html
```

### WEBSPEX
Hate to build?
Visit our webservice [WEBSPEX](https://auspex.physnet.uni-hamburg.de/)

