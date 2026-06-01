# Global-Local Hierarchical Alignment and Semantic Calibration Fusion for Multimodal Sentiment Analysis

## Usage

### Pre-requisites
- Python 3.9.13
- PyTorch 1.13.0
- CUDA 11.7

### Installation
- conda create -n XXX(your environment) python==3.9.13
```
- Activate the built XXX environment.
```
- conda activate xxx
```
- Install Pytorch with CUDA
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0


### Datasets
Data files can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing). 
You can first build and then put the downloaded datasets into `./dataset` directory and revise the path in `./config/config.json`. For example, if the processed the MOSI dataset is located in `./dataset/MOSI/aligned_50.pkl`. Please make sure "dataset_root_dir": "./dataset" and "featurePath": "MOSI/aligned_50.pkl". For more details, please follow the [official website](https://github.com/ecfm/CMU-MultimodalSDK) of these datasets.
Besides, it is recommended to manually download the pre-trained BERT model from the official website and add it to `bert-base-uncased`.

### Run the Codes
- Training

You can first set the training dataset name in `./train.py` as "mosei" or "mosi", and then run:
```
run train.py
```
By default, the trained model will be saved in `./pt` directory. You can change this in `train.py`.

- Testing

You can first set the testing dataset name in `./test.py` as "mosei" or "mosi", and then test the trained model:
```
run test.py

```
### Thanks
The entire project borrowed from the coding approach of [DLF](https://github.com/pwang322/DLF). The operating environment is basically the same, and we also reference and compare the baseline of that work.
