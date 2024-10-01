# UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation [MM 2024]

This repository contains the implementation of our manuscript titled "[UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation](https://arxiv.org/pdf/2404.14241.pdf)", which has been accepted for publication at the ACM Multimedia Conference 2024.

As the codebase is extensive and complex, this repository will be actively maintained and updated. The dataset is currently being refined due to its large size and will be released on Hugging Face shortly.

## Usage

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

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zhong2024urbancross,
  title={UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation},
  author={Zhong, S. and Hao, X. and Yan, Y. and others},
  booktitle={Proceedings of the ACM Multimedia Conference},
  year={2024},
  note={Accepted}
}