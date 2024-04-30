# Sophisticated Learning Algorithm Implementations

This repository hosts the source code for the "Sophisticated Learning" algorithm as described in the research paper:

> Hodson, et al. "Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning."

The codebase includes implementations for all four algorithms presented in the paper: Sophisticated Learning, Sophisticated Inference, Bayes-Adaptive RL, and Bayes-Adaptive RL + UCB. Each MATLAB script within the 'Algorithms' directory is named according to the algorithm it implements.

## Authors

This code was developed by:
- Rowan Hodson @RowanLIBR
- St John Grimbly @sgrimbly
- Evert Boonstra @eboonst

## Usage

To run one simulation of 120 iterations of any algorithm, navigate to the 'matlab' directory and execute the `main.m` script. Set `run_SL`, `run_SI`, `run_BA`, or `run_BAUCB` to 1 to activate Sophisticated Learning, Sophisticated Inference, Bayes-Adaptive RL, and Bayes-Adaptive RL + UCB, respectively.
 
## Description

The repository contains MATLAB and Python implementations intended for reproducing the research results and further development.

### Directory Structure

## Getting Started

### Prerequisites

- MATLAB (R2018b or newer recommended)
- Python 3.6+ (with virtual environment recommended)

### Installation

#### MATLAB
Ensure MATLAB is installed. The scripts should run in any standard MATLAB environment.

#### Python
Set up a Python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r python/requirements.txt
```

## Usage

### Running the MATLAB Scripts
Navigate to the `matlab` folder and execute the scripts:

```matlab
cd matlab
main
```

### Running the Python Scripts
Ensure you are in the virtual environment and run the Python scripts:

```bash
cd python
python main.py
```

## Examples

Here are a few quick examples to get started:

### MATLAB
```matlab
% Load data
data = load('example_data.mat');
% Run algorithm
results = sophisticated_learning(data);
```

### Python
```python
# Load data
data = load_data('example_data.json')
# Run algorithm
results = sophisticated_learning(data)
```

## Citation

If you use this code for your research, please cite it as follows:

```bibtex
@misc{hodson2023planning,
      title={Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning}, 
      author={Rowan Hodson and St John Grimbly and Evert Boonstra and Bruce Bassett and Charel van Hoof and Benjamin Rosman and Mark Solms and Jonathan P. Shock and Ryan Smith},
      year={2023},
      eprint={2308.08029}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This code is provided "as is", without warranty of any kind, express or implied. Please note that the MATLAB version may have dependencies on specific toolboxes.

## Contact

For any questions or concerns, please open an issue on GitHub or contact the corresponding author of the associated papers.
