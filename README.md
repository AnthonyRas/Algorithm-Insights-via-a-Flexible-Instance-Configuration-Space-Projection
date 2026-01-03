# Algorithm-Insights-via-a-Flexible-Instance-Configuration-Space-Projection

## Overview

This repository provides the source code for the Instance-Configuration Analysis (ICSA) toolkit. This extends Instance Space Analysis (ISA) for the analysis of algorithm performance in terms of both instance features and algorithm parameters.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnthonyRas/Algorithm-Insights-via-a-Flexible-Instance-Configuration-Space-Projection.git
   cd Algorithm-Insights-via-a-Flexible-Instance-Configuration-Space-Projection
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv/Scripts/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**CAGen**, an external tool, is used for generating covering arrays (web interface: https://srd.sba-research.org/tools/cagen/).

## Repository Structure

```
├── icsa.py                      # Main ICSA class
├── projector.py                 # Neural network projection models
├── preprocessor.py              # Data preprocessing utilities
├── plotter.py                   # Interactive visualisation (Panel + Plotly)
├── space_updates.py             # Plot update coordination
├── median_plots.py              # Notched boxplot utilities
├── experimental_design.py       # Covering array sampling
├── classification_icsa.py       # Classification-specific utilities
├── network_analysis.py          # Network architecture experiments
│
├── 00EXP_classification_icsa_main.ipynb  # Main experiment notebook
├── demo.ipynb                            # Interactive toolkit demo
├── network_experiments.ipynb             # Architecture experiments
├── network_experiments_analysis.ipynb    # Architecture analysis
├── knn_IG_boxplots.ipynb                # KNN instance group analysis
│
├── analysis/                    # Generated data and results
│   └── classification/
│       ├── knn/                 # KNN experiment files
│       ├── sgdclassifier/       # SGDC experiment files
│       ├── instances/           # Classification datasets (.arff) from MATILDA (https://matilda.unimelb.edu.au/matilda/our-methodology)
│       └── features/            # Instance feature data
│
├── Figures/                     # Supplementary figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

To launch the interactive ICSA toolkit with pre-computed results, run the demo.ipynb notebook. The fit_overlapping function produces the visual toolkit if results are already computed. Otherwise, it trains the projection models first.

### Basic Usage Example

```python
import panel as pn
from icsa import ICSA, set_seeds
from classification_icsa import load_knn_metadata, MCC

# Initialise Panel with Plotly support (required for visualisation)
pn.extension('plotly')

# Set random seed for reproducibility
set_seeds(0)

# Load metadata (features, parameters, performance)
knn_fpath = 'analysis/classification/knn/'
metadata = load_knn_metadata(knn_fpath, confusion_function=MCC)

# Create ICSA analysis with overlay mode
analysis = ICSA(
    optimisation_direction='max',  # MCC should be maximised
    mode='overlay',            # or 'concatenate'
    performance_metric_name='MCC'
)

# Fit and show interactive visualisation
analysis.fit_show(
    model_fpath=knn_fpath + 'mcc/overlapping/',
    metadata=metadata,
    epochs=150,
    validation_split=0.1,
    save=True  # Cache results for faster subsequent runs
)
```

**Note**: Run this in a Jupyter notebook for the interactive visualisation to display properly.

## Reproducing the Paper Results

This was originally tested with Python 3.10.

Firstly, delete all files in "knn" and/or "sgdclassifier" and all subfolders, but keep the folders.

Run the notebook 00EXP_classification_icsa_main.ipynb to reproduce the paper's results. When running this notebook, a parameter_space.txt file is generated, which is to be imported into the web interface of **CAGen**. Importantly, the parameter constraints should be written into **CAGen**'s constraints field; these constraints are given as comments on the notebook.

Go to the "Generate" tab and select the value of t (4 for KNN, 3 for SGDC). Click "Randomize Don't-Care Values" and then export the covering array as a CSV file. Place it in analysis/classification/[classifier name]/.

Then, run the block containing "get_param_configs" and then "run_classifier_configs". After this is finished, you can run demo.ipynb, which will train the projection models and produce the visualisations.

The toolkit provides:
* Lasso selection of instances/configurations
* Dynamic performance aggregation over selections
* Feature/parameter visualisation via colour scales
* Boxplots and notched median plots
* Configuration tables with performance rankings
* Instance feature heatmaps (percentile ranks)

To reproduce the experiments on network architectures, simply run the network_experiments.ipynb notebook and then run all the blocks in network_experiments_analysis.ipynb.

## Citation

TBD

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This research was supported by an Australian Government Research Training Program (RTP) Scholarship, through a dual-award program between The University of Melbourne and The University of Manchester. This research was (fully/partially) funded by the Australian Government through the Australian Research Council Industrial Transformation Training Centre in Optimisation Technologies, Integrated Methodologies, and Applications (OPTIMA), Project ID IC200100009.


