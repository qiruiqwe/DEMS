# A Linear Genetic Programming Method for Adaptive and Interpretable Energy Scheduling

## Introduction

In modern energy scheduling systems, achieving adaptive and interpretable control is a significant challenge. This paper proposes a novel approach using Linear Genetic Programming (LGP) to address the problem of action coupling in complex energy scheduling environments. Our method evolves interpretable control policies that can effectively manage interdependencies between actions, leading to more robust and efficient scheduling solutions.

This repository provides the source code to reproduce the experiments and results presented in our paper.

## Features

-   **Linear Genetic Programming Implementation**: A flexible and powerful LGP core for evolving control strategies.
-   **Energy Scheduling Environment**: A simulated environment that captures the complexities of action coupling in energy systems.
-   **Adaptive Control**: The evolved models demonstrate strong adaptability to dynamic changes in the energy environment.
-   **Interpretability**: The resulting LGP-based controllers are inherently interpretable, allowing for clear analysis of the learned scheduling policies.
-   **Reproducibility**: All necessary code and instructions are provided to replicate the findings of our study.

## Runtime Environment

The code is written in Python 3.9. 

| Component   | Version (minimum) | Notes                                      |
|-------------|-------------------|--------------------------------------------|
| Python      | 3.9.x             | Not tested on ≤ 3.8                        |
| PyTorch     | 2.1.0             | Both CPU & CUDA 11.8 wheels supported      |
| Gymnasium   | 0.29.0            | Official successor to OpenAI Gym           |
| Numba       | ≥ 0.60.0          | Accelerates core numerical loops           |
| pybind11    | ≥ 2.12            | Required only if re-compiling C++ extensions|
| matplotlib  | latest            | Used for plotting results                  |

## One-line Installation
```bash
conda create -n myenv python=3.9 -y && conda activate myenv
pip install torch==2.1.0 gymnasium==0.29.0
pip install "numba>=0.60.0" "pybind11>=2.12" matplotlib
