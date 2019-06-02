# ScenarioMeta
### [Project](https://sites.google.com/view/scenariometa) | Arxiv

Sequential Scenario-Specific Meta Learner for Online Recommendation

Zhengxiao Du, Xiaowei Wang, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jingren Zhou](http://www.cs.columbia.edu/~jrzhou/), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/)

Accepted to KDD 2019 Applied Data Science Track!

Under construction. Expect an official release of the dataset and cleaner code in June

## Prerequisites

- Python 3
- PyTorch >= 1.0.0
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/ScenarioMeta
cd ScenarioMeta
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

Three public datasets are used for experiments. The Taobao Cloud Theme Click Dataset is released by us.

- Amazon Review [Source](http://jmcauley.ucsd.edu/data/amazon)
- Movielens-20M [Source](https://grouplens.org/datasets/movielens/20m/)
- Taobao Cloud Theme Click Dataset [Source](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9716)

You can download the preprocessed datasets from OneDrive by running `python scripts/download_preprocessed_data.py`.
If you're in regions where OneDrive is not available (e.g. Mainland China), try to use Tsinghua Cloud by running `python scripts/download_preprocessed_data.py --tsinghua`.

### Training

Different configurations for datasets in the paper are stored under the `configs/` directory. Launch a experiment with `--config` to specify the configuration file, `--root_directory` to specify the path to the downloaded data, `--comment` to specify the experiment name which will be used in logging and `--gpu` to speficy the gpu id to use. 

## Cite