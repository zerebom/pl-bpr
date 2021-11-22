# pl-bpr

pl-bpr is experimental code for comparing Bayesian personalization ranking and Matrix factorization, implemented in Pytorch-lightning.
 
# Requirement
  
```bash
pytorch                   1.8.0           cpu_py38hc43b888_1    conda-forge
pytorch-lightning         1.5.2              pyhd8ed1ab_0    conda-forge
torch                     1.9.0                    pypi_0    pypi
torchmetrics              0.5.1              pyhd8ed1ab_0    conda-forge
wandb                     0.12.2                   pypi_0    pypi
numpy                     1.20.3                   pypi_0    pypi

```

</details>

 
# Usage
 
 
```bash
git clone https://github.com/zerebom/bpr
cd bpr
python main.py
```
 
# Reference
Implement: https://github.com/EthanRosenthal/torchmf

Paper: 
```
@article{rendle2012bpr,
  title={BPR: Bayesian personalized ranking from implicit feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1205.2618},
  year={2012}
}
```
