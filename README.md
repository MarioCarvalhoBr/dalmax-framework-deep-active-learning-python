# DalMax - Framework to Deep Active Learning Approaches
Repository of resources used in my doctoral studies in the area of ​​Deep Active Learning

![Deep Active Learning Framework](assets/active-learning-framework.png)

## Methodology Overview
The Framework to Deep Active Learning Approaches for Maximal Information Gain (DalMax) is a framework that aims to select the most informative samples for training a deep learning model. The DalMax framework is based on heuristic strategies that select the samples that maximize the information gain of the model. The DalMax framework is composed of this heuristic strategies:
  
- **Random Sampling**: Select samples randomly.
- **Least Confidence**: Select samples where the model is least confident[1].
- **Margin Sampling**: Select samples where the margin between the two most likely classes is smallest[2].
- **Entropy Sampling**: Select samples where the entropy of the prediction is highest[3].
- **Uncertainty Sampling with Dropout Estimation**: Select samples where the model is most uncertain. This strategy uses dropout to estimate the uncertainty of the model[4].
- **Bayesian Active Learning Disagreement**: Select samples where the model is most uncertain. This strategy uses Bayesian methods to estimate the uncertainty of the model[4].
- **Cluster-Based Selection**: Select samples that are most representative of the clusters[5].
- **Adversarial Margin**: Select samples where the adversarial margin is smallest[6].

- 
## Dependencies
This project depends on the following libraries:
- Python==3.9
- CUDA==12.4
- Pytorch==v2.5.0

## Environment Setup
To install the necessary dependencies, run the following command:

 - CONDA: Create a virtual environment with Python 3.9.
    ```bash
        conda create --name dalmax_gpu python=3.9
        conda activate dalmax_gpu
    ```
 - VENV: Create a virtual environment with Python 3.9.
    ```bash
        python3 -m venv dalmax_gpu
        source tf/bin/activate   
    ```

#### Dependencies
 - Install the dependencies on `requirements.txt`.
    ```bash
        pip install -r requirements.txt
    ```
 - Or install the dependencies manually.
    ```bash
        matplotlib==3.9.2
        numpy==2.1.3
        Pillow==11.0.0
        scikit_learn==1.5.2
        torch==2.5.0
        torchvision==0.20.0
        tqdm==4.67.1

    ```
    - Or Conda install the dependencies manually.
        ```bash
            conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
        ```

## Usage

### Dataset
This project uses the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
Download the dataset zip CIFAR10 from [https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing](https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing). It must be unzipped inside the `DATA` folder.

#### Steps dataset
 - Create folder DATA in root of project.
    ```bash
        mkdir DATA
    ```
 - Download the dataset zip CIFAR10 from [https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing](https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing).
 - Unzip the dataset inside the `DATA` folder.
    ```bash
        unzip DATA_CIFAR10.zip -d DATA
    ```

Structure of the dataset:
```
DATA/
    DATA_CIFAR10/
        train/
            0/
                1000.png
                1001.png
                ...
            1/
            2/
            ...
            8/
            9/
        test/
            0/
                1000.png
                1001.png
                ...
            1/
            2/
            ...
            8/
            9/
```

### Training Deep Active Learning Models
To train the models, run the following command:
```bash
    python demo.py \
        --n_round 10 \
        --n_query 1000 \
        --n_init_labeled 10000 \
        --dataset_name DANINHAS \
        --params_json params.json \
        --strategy_name RandomSampling \
        --seed 1
```

#### Parameters
- `n_round`: Number of iterations.
- `n_query`: Number of samples to be selected in each iteration.
- `n_init_labeled`: Number of labeled samples in the initial training.
- `dataset_name`: Name of the dataset. Options: `CIFAR10`, `DANINHAS`. If not informed, the dataset is `CIFAR10`.
- `strategy_name`: Name of the active learning strategy. Options: `RandomSampling`, `LeastConfidence`, `MarginSampling`, `EntropySampling`, `LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`, `KMeansSampling`, `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`, `AdversarialDeepFool`. If not informed, the strategy is `RandomSampling`.
- `seed`: Seed for random number generation. Optional. If not informed, the seed is randomly generated.
- `img_size`: Image size. Optional. If not informed, the image size is 32x32. Example --img_size 64.
- `params_json`: JSON file with the hyperparameters of the model. If not informed, error occurs. Example --params_json params.json.

##### Hyperparameters
The hyperparameters of the model are defined in a JSON file. The JSON file must have the following structure:
```json
{
    "YOU_DATASET_NAME": {
        "n_epoch": 10,
        "n_drop": 5,
        "n_classes": 10,
        "train_args": { "batch_size": 256, "num_workers": 4 },
        "test_args": { "batch_size": 256, "num_workers": 4 },
        "optimizer_args": { "lr": 0.05, "momentum": 0.3 }
    },
    "CIFAR10": {
        "n_epoch": 20,
        "n_drop": 10,
        "n_classes": 10,
        "train_args": { "batch_size": 64, "num_workers": 1 },
        "test_args": { "batch_size": 1000, "num_workers": 1 },
        "optimizer_args": { "lr": 0.05, "momentum": 0.3 }
    },
    "DANINHAS": {
        "n_epoch": 10,
        "n_drop": 5,
        "n_classes": 5,
        "train_args": { "batch_size": 256, "num_workers": 4 },
        "test_args": { "batch_size": 256, "num_workers": 4 },
        "optimizer_args": { "lr": 0.05, "momentum": 0.3 }
    }
}

```
## License

DalMax is under the MIT License. See the [LICENSE](LICENSE) file for more details.


## Citing

If you use our code in your research or applications, please consider citing our repository.

```
@misc{Carvalho2024dalmax,
  author       = {Mário de Araújo Carvalho},
  title        = {DalMax: Framework for Deep Active Learning Approaches},
  howpublished = {\url{https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python}},
  month        = {dec},
  year         = {2024},
  note         = {Available on GitHub},
  annote       = {A Python framework for implementing and comparing deep active learning methods.}
}

```

## Contact

- Author: Mário de Araújo Carvalho
- Email: mariodearaujocarvalho@gmail.com
- Project: [https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python](https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python)


## Reference

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009

[4] Deep Bayesian Active Learning with Image Data, ICML, 2017

[5] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[6] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018


## Extras

This project is based on some code from the repository [DeepAL: Deep Active Learning in Python](https://github.com/ej0cl6/deep-active-learning). Additionally, please consider citing the corresponding work. For more details, refer to the publication available [here](https://arxiv.org/abs/2111.15258). 
