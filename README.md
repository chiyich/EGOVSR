
# EgoVSR: Towards High-Quality Egocentric Video Super-Resolution

## Paper
EgoVSR: Towards High-Quality Egocentric Video Super-Resolution

Yichen Chi, Junhao Gu, Jiamiao Zhang, Wenming Yang, Yapeng Tian

[Arxiv](https://arxiv.org/abs/2305.14708)

./Egovsr_demo.mp4

## Installation
This work is based on the MMEditing (now MMagic) framework. Thanks to OpenMMLab.

MMEditing depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install MMEditing from source.

```shell
git clone https://github.com/chiyich/EGOVSR.git
cd EGOVSR
pip3 install -e .
```

Please refer to [install.md](docs/en/install.md) for more detailed instruction.

## Getting Started

**Step 1.**
Download our [checkpoint and EGOVSR test/valid dataset](https://drive.google.com/drive/folders/1yjlvGVUb8F8KsGYrzMeQOO3fcMWsvIQS?usp=sharing) 
and  [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html).
We need **train_sharp** subset and **train_blur** if you need to train second-order model.

**Step 2.**
Prepare datasets and modify the folder location in config files.

**Step 3.**
Train your own model(4 indicates the number of GPUs):
```shell
#for first stage training (L1 Model)
bash tools/dist_train.sh configs/egovsr/egovsr_L1_reds.py 4
#for second stage training (GAN Model)
bash tools/dist_train.sh configs/egovsr/egovsr_reds.py 4
```
Or test:
```shell
python tools/test.py configs/egovsr/egovsr_reds.py experiments/egovsr/iter_250000.pth --save-path work_dirs/results/
```

## Contributing

We appreciate all contributions to improve MMEditing. Please refer to our [contributing guidelines](https://github.com/open-mmlab/mmediting/wiki/A.-Contribution-Guidelines).



## Citation

If our work is helpful to your research, please cite it as below.

```bibtex
@article{chi2023egovsr,
  title={EgoVSR: Towards High-Quality Egocentric Video Super-Resolution},
  author={Chi, Yichen and Gu, Junhao and Zhang, Jiamiao and Yang, Wenming and Tian, Yapeng},
  journal={arXiv preprint arXiv:2305.14708},
  year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
