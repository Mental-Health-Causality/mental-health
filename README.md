<div align="center">

# Causal Machine Learning for Mental Health Research

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

# TODOS

- [ ] Add description of the project
- [ ] Add Notebook descriptions
- [ ] Add final article

## Description

Everything is in the [notebooks](notebooks) folder.

Each notebook is a separate experiment. The tldr of  each notebobook is:

- [notebooks/XXXXXXXX.ipynb](notebooks/XXXXXXXX.ipynb) - Add Description Here
- [notebooks/baseline.py](notebooks/baseline.py) - classification model which uses the causal features and all features
- [notebooks/data_transformation_IRT.ipynb](notebooks/data_transformation_IRT.ipynb) - Apply IRT to transform categorical features `suicide` and `anxiety` to continuos
- [notebooks/interpretability/shap_sample_captum_markov_blanket_features.ipynb](notebooks/interpretability/shap_sample_captum_markov_blanket_features.ipynb) - Load the classification model which uses the causal fetures and then apply SHAP using _Captum_ lib
- [notebooks/interpretability/shap_sample_captum_all_features.ipynb](notebooks/interpretability/shap_sample_captum_all_features.ipynb) - Load the classification model which uses all features and then apply SHAP using _Captum_ lib


## Final Report

- [Final Report (PDF)](final_report.pdf)

## Final Presentation

- [Final Presentation (PDF)](final_report_presentation.pdf)


## Related Papers Presentation

[![Paper Presentation](https://img.youtube.com/vi/tt1ReJAr6tM/0.jpg)](https://youtu.be/tt1ReJAr6tM)


<!-- ## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
``` -->
