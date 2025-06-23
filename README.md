
# Mixture-of-Experts Time Series Forecasting

This repository provides frameworks for implementing Mixture-of-Experts (MoE) models tailored for time series forecasting tasks. Leveraging the MoE architecture allows for efficient handling of diverse time series data by activating a subset of specialized experts for each input.

## Introduction

Deep learning is increasingly used for time series forecasting. N-BEATS, built from stacks of multilayer perceptron blocks, has achieved state-of-the-art results and offers interpretability by decomposing forecasts into components like trend and seasonality.

This project introduces N-BEATS-MOE, an extension that adds a Mixture-of-Experts (MoE) layer with a gating network. This allows the model to better adapt to each time series and potentially improves interpretability by showing which expert is most relevant.

We evaluate N-BEATS-MOE on 12 benchmark datasets, showing consistent improvements, especially on heterogeneous time series.



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Correspondence](#model-correspondence)
- [Citations](#citations)
- [Contributing](#contributing)
- [License](#license)

## Overview

Mixture-of-Experts (MoE) models are a class of machine learning architectures that utilize a gating mechanism to route inputs to a subset of specialized experts. In the context of time series forecasting, MoE models can effectively capture temporal patterns and dependencies by activating relevant experts based on the input sequence.

This repository offers implementations of MoE frameworks specifically designed for time series forecasting tasks. By employing MoE architectures, users can build models that are both computationally efficient and capable of capturing complex temporal dynamics.

## Features

- **Modular Architecture**: Easily extendable components for defining experts, gating mechanisms, and routing strategies.
- **Scalability**: Designed to handle large-scale time series datasets efficiently.
- **Flexibility**: Supports various configurations of experts and gating mechanisms to suit different forecasting tasks.
- **Integration**: Compatible with popular machine learning frameworks, facilitating seamless integration into existing workflows.

## Repository Structure

```
mixture_of_experts_time_series/
├── conf/                # Configuration files
├── datasets/            # Time series datasets
├── db/                  # Database utilities
├── models/              # Model definitions
├── nbs/                 # Jupyter notebooks for experimentation
├── pgfs/                # LaTeX PGFPlots for visualizations
├── results/             # Output and results of experiments
├── .gitignore           # Git ignore file
├── README.md            # This README file
├── get_data.py          # Script to fetch datasets
├── requirements.txt     # Python dependencies
├── results_summary_with_blcs.csv  # Summary of results
├── run_exp.py           # Script to run experiments
├── run_hyper.py         # Script for hyperparameter tuning
├── run_model.py         # Script to train and evaluate models
├── todo.md              # To-do list for future developments
└── utils.py             # Utility functions
```

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run experiments using the provided scripts, you can utilize the following commands:

- Run a single model:

```bash
python run_model.py
```

- Execute a predefined experiment:

```bash
python run_exp.py
```

- Perform hyperparameter tuning:

```bash
python run_hyper.py
```

For more detailed usage and examples, refer to the Jupyter notebooks located in the `nbs/` directory. Make sure to include the necessary configurations in the `conf/` directory to customize your experiments.

## Model Correspondence

| Python Class Name | Paper Model Name | Description                                     |
|------------------|-------------------|-------------------------------------------------|
| N-BEATS-MOE      | `NBeatsStackMoe`        | N-BEATS extended with Mixture-of-Experts layer (proposed approach)  |
| NBeatsMoe    | `MoEBlock`             | Mixture-of-Experts gating and expert module     |
| NBeatsMoe  | `MoEShared`          | Shared MoE (set share_experts=true) |
| NBeatsMoe  | `MoEScaled`          | Scaled MoE (set scale_expert_complexity=true) |

*Note:* Class names correspond to components described in the associated research paper and dissertation.

There are also several variations available that we don't discuss in the paper. For example, in the N-BEATS-MOE model, you might want to test different activation functions for the gating network, such as `Sigmoid`. 


---

## Citations

If you use this work, please consider citing:

```bibtex

@inproceedings{matos2025nbeatsmoe,
  title={{N-BEATS-MOE}: N-BEATS with a Mixture-of-Experts Layer for Heterogeneous Time Series Forecasting},
  author={Matos, Ricardo and Roque, Luis and Cerqueira, Vitor},
  booktitle={TODO},
  year={2025},
  organization={University of Porto},
  address={Porto, Portugal}
}


```

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
