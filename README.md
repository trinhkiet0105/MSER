
<h1 align="center">
  SER_ICIIT_2024
  <br>
</h1>

<h4 align="center">Official code repository for paper "SER_ICIIT_2024"</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/SER_ICIIT_2024?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/SER_ICIIT_2024?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/SER_ICIIT_2024?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> Some abstract here
## Key Features
- SER_ICIIT_2024 - some description here
## How To Use
- Clone this repository 
```bash
git clone https://github.com/namphuongtran9196/SER_ICIIT_2024.git 
cd SER_ICIIT_2024
```
- Create a conda environment and install requirements
```bash
conda create -n SER_ICIIT_2024 python=3.8 -y
conda activate SER_ICIIT_2024
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
or (only work for Debian Bookworm):
```bash
conda env create -f environment.yml
conda activate SER_ICIIT_2024
```

- There are some error with torchvggish when using with GPU. Please change the code at line 58,59 in file torchvggish/model.py to accept the GPU (this file may locate on your sites-packages in Python). I will push a fix source code for this repo later.
```python
        self._pca_matrix = torch.as_tensor(params["pca_eigen_vectors"]).float().cuda()
        self._pca_means = torch.as_tensor(params["pca_means"].reshape(-1, 1)).float().cuda()
        # Or you can set somethings like .to(device) to make it flexible
```
- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). Or you can download our preprocessed dataset [here](https://drive.google.com/file/d/10mAGxugX9LZ12MVW_P0x7PG009acygGh/view?usp=sharing).

- Download dataset from [here](https://www.kaggle.com/datasets/jamaliasultanajisha/iemocap-full) and extract it to the data folder. The data folder should look like this:
```
scripts/notebooks
                ├── data
                │   ├── IEMOCAP
                │   │   ├── train.pkl
                │   │   ├── test.pkl
                ├── IEMOCAP_full_release
                ├── train.py/train.ipynb
```

- Preprocess data
```bash
cd scripts && python preprocess.py --data_root <path_to_iemocap_dataset> --output_dir <path_to_output_folder>
```

- Before starting training, you need to modify the [config file](./src/configs/bert_vggish.py) in the config folder. You can refer to the config file in the config folder for more details.
```bash
cd scripts && python train.py -cfg <path_to_config_file>
```
- You can also use the [notebook file](notebooks/train.ipynb) to train the model. Just open the notebook file and run it.

## Download
## License

## Citation
If you use this code or part of it, please cite our work. On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository" button. Or, see the comments in CITATION.cff for the raw BibTeX.

## References
---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;
