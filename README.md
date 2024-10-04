# UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation [MM 2024]

This repository contains the implementation of our manuscript titled "[UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation](https://arxiv.org/pdf/2404.14241.pdf)", accepted for publication at ACM Multimedia 2024. 

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Dataset](#dataset)
- [Citation](#citation)
- [Contact](#contact)

## Overview
UrbanCross aims to enhance the performance of satellite image-text retrieval tasks by addressing the domain gaps that arise from diverse urban environments. The framework incorporates:


![framework](/figs/framework.png)

- A cross-domain dataset enriched with geo-tags across multiple countries.
- Large Multimodal Model (LMM) for textual refinement and Segment Anything Model (SAM) for visual augmentation.
- Adaptive curriculum-based sampling and weighted adversarial fine-tuning modules.

As the codebase is extensive and complex, this repository will be actively maintained and updated. The dataset is currently being refined due to its large size and will be released on Hugging Face shortly.

## Usage

### Prerequisites
- Python 3.8+
- PyTorch 1.10+ with CUDA support
- Other dependencies listed in `requirements.txt`

You can install the required Python packages using:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a Conda environment with:

```shell
conda create -n urbancross python=3.8
conda activate urbancross
pip install -r requirements.txt
```

### Run

For instructions on how to run the code, please refer to the `cmd` directory for the respective shell scripts.

```shell                 
.
├── fine-tune
│   ├── finetune_urbancross_curriculum.sh
│   ├── finetune_urbancross.sh
│   └── zeroshot_urbancross.sh
├── test
│   ├── test_urbancross_finland.sh
│   ├── test_urbancross_germany.sh
│   ├── test_urbancross_rsicd.sh
│   ├── test_urbancross_rsitmd.sh
│   ├── test_urbancross_spain.sh
│   ├── test_urbancross_without_sam_finland.sh
│   ├── test_urbancross_without_sam_germany.sh
│   ├── test_urbancross_without_sam_integration.sh
│   ├── test_urbancross_without_sam_rsicd.sh
│   ├── test_urbancross_without_sam_rsitmd.sh
│   └── test_urbancross_without_sam_spain.sh
└── train
    ├── train_urbancross_finland.sh
    ├── train_urbancross_germany.sh
    ├── train_urbancross_rsicd.sh
    ├── train_urbancross_rsitmd.sh
    ├── train_urbancross_spain.sh
    ├── train_urbancross_without_sam_finland.sh
    ├── train_urbancross_without_sam_germany.sh
    ├── train_urbancross_without_sam_integration.sh
    ├── train_urbancross_without_sam_rsicd.sh
    ├── train_urbancross_without_sam_rsitmd.sh
    └── train_urbancross_without_sam_spain.sh
```

## Dataset
The dataset used in UrbanCross is currently being refined and will be released on Hugging Face soon. It includes:

- High-resolution satellite images from multiple countries.
- Geo-tags to enhance retrieval performance
- Text descriptions generated through a multimodal model.

![dataset](/figs/dataset.png)


## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhong2024urbancross,
  title={UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation},
  author={Zhong, Siru and Hao, Xixuan and Yan, Yibo and Zhang, Ying and Song, Yangqiu and Liang, Yuxuan},
  journal={arXiv preprint arXiv:2404.14241},
  year={2024}
}
```

## Contact
For any questions or issues, feel free to open an issue or contact the authors:

- Siru Zhong: siruzhong@outlook.com
- Yuxuan Liang (Corresponding Author): yuxliang@outlook.com
