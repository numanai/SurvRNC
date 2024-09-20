# SurvRNC: Learning Ordered Representations for Survival Prediction using Rank-N-Contrast

<p align="center">
    <img src="./docs/SurvRNC.png" alt="Image" width="35%" height="35%">
</p>

> [**SurvRNC: Learning Ordered Representations for Survival Prediction using Rank-N-Contrast**](https://arxiv.org/pdf/2403.10603) <br>
> [Numan Saeed](https://numanai.github.io/)* , [Muhammad Ridzuan](https://mfarnas.github.io/ridzuan-healthcare-ai/)* , Fadillah Adamsyah Maani, Hussain Alasmawi, Karthik Nandakumar, Mohammad Yaqub

\* Equally contributing first authors

**Mohamed bin Zayed University of Artificial Intelligence**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2403.10603)
[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://hecktor.grand-challenge.org/)




Official GitHub repository for the SurvRNC

---


# SurvRNC

SurvRNC is a project focused on survival analysis using Rank-N-Contrast loss to order the latent representation for prognosis. It restricts the latent representation of both uni/multi-modal data to be ordered based on the time-to-event.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration Management](#configuration-management)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

Please refer to the [Installation Guide](./docs/Installation.md) for detailed setup instructions.

## Usage

For detailed usage instructions, please see the [Usage Guide](./docs/Usage.md).

## Configuration Management

The SurvRNC project utilizes configuration files to manage various parameters for different scripts. This approach enhances organization, scalability, and reproducibility.

### Configuration Files

Configuration files are stored in the `configs/` directory and use the YAML format. The `default.yaml` file contains default settings, which can be overridden by specific configuration files like `train.yaml` or `preprocess.yaml`.

### Using Configuration Files

#### Preprocessing Data

```bash
python ctpt_preprocess.py --config configs/preprocess.yaml
```

To override specific parameters from the command line:

```bash
python ctpt_preprocess.py --config configs/preprocess.yaml --override space_x=3 batch_size=64
```

#### Training the Model

```bash
python main.py --config configs/train.yaml
```

To override specific parameters:

```bash
python main.py --config configs/train.yaml --override lr=0.001 epochs=100
```

### Creating New Configuration Files

You can create new configuration files by copying `default.yaml` and modifying the required parameters. Ensure that these files are stored in the `configs/` directory for consistency.

### Parameter Overrides

The `--override` argument allows you to specify individual parameter changes without editing the configuration file. Parameters should be provided in the `key=value` format. Nested parameters can be accessed using dot notation if necessary.

**Example:**

```bash
python main.py --config configs/train.yaml --override optimizer=SGD lr=0.005
```
## Data Preprocessing

The `ctpt_preprocess.py` script is used to preprocess CT and PT scans. It now uses configuration files for parameter management.

```bash
python ctpt_preprocess.py --config configs/preprocess.yaml
```

## Training

Run the main training script with the appropriate configuration file:

```bash
python main.py --config configs/train.yaml
```


## Evaluation

Evaluate the trained models using survival analysis metrics such as Concordance Index (CI) and Brier Score. The evaluation process is integrated into the training script and uses the same configuration file system.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Citation
If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:

```bibtex
@article{saeed2024survrnc,
  title={SurvRNC: Learning Ordered Representations for Survival Prediction using Rank-N-Contrast},
  author={Saeed, Numan and Ridzuan, Muhammad and Maani, Fadillah Adamsyah and Alasmawi, Hussain and Nandakumar, Karthik and Yaqub, Mohammad},
  journal={arXiv preprint arXiv:2403.10603},
  year={2024}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at numan.saeed@mbzuai.ac.ae.

