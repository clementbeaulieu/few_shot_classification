# Classification Few-Shot

Research Paper. Master in Management HEC Paris, Major in Managerial and Financial Economics.

This deep learning project provides a scalable framework for Meta-Learning architecture in a supervised environment, for few-shot classification. It provides Model-Agnostic Meta-Learning and Prototypical frameworks. It compares those methods to conventional Machine Learning. Several applications of those architectures are implemented to test their efficiency and performance.

This project is higly inspired from the following respositories :

https://github.com/fmu2/PyTorch-MAML

https://github.com/dragen1860/MAML-Pytorch

https://github.com/cbfinn/maml

## Mini-ImageNet Dataset.

Mini-ImageNet is a variation of ImageNet dataset introduced by authors in their paper [Matching Networks for One-Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) and is downlable from the following link, thanks to the courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting) :

[mini-ImgeNet dataset](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view)

## Meta-Learning Architectures.

- Model-Agnostic Meta-Learning, architecture introduced by [Model-Agnostic Meta-Learning paper](https://arxiv.org/abs/1703.0340) and the original Tensorflow implementation [MAML Tensorflow implementation](https://github.com/cbfinn/maml).

- Deep Sets Equivariant architecture.

## Models.

- ConvNet4 encoder
- ResNet12 encoder

## Project structure

The project is structured as following:

```bash
.
├── loaders
|  └── dataset selector
|  └── mini_imagenet.py # loading and pre-processing mini_imagenet meta data
|  └── transforms.py # image transformations
├── models
|  └── architecture selector
|  └── encoders # folder for encoders construction
|  └── classifiers # folder for classifiers construction
|  └── modules.py # Batchnorm, Convolutional layers definition, that are scalable for meta-training, validation and testing
|  └── models.py # models creations built with an encoder + a classifier
|  └── maml.py # MAML model
|  └── prototypical_net.py # Prototypical Networks model
├── utils
|  └── losses.py  # loss selector
|  └── optimizers.py  # optimizer selector
├── run.py # main file from the project serving for calling all necessary functions for (meta-)training and (meta-)testing
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
```

## Launching
Experiments can be launched by calling `run.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python3 run.py --root-dir /content/drive/MyDrive/data --config /content/few_shot_classification/configs/convnet4/mini_imagenet/5_way_1_shot/reproduce_maml.yaml --name 5_way_2_shot_batch100_epoch50 --gpu=0,1,2,3 --meta --train-n_way 5 --train-n_shot 2 --val-n_way 5 --val-n_shot 2 --test-n_way 5 --test-n_shot 2  --tensorboard --val`
  + this experiment is on the _mini_imagenet_ dataset which can be found in `--root-dir /content/drive/MyDrive/data` trained over _maml_ architecture. It optimizes with _adam_ with general training conditions found in config `/content/few_shot_classification/configs/convnet4/mini_imagenet/5_way_1_shot/reproduce_maml.yaml`. In addition it saves intermediate results to `--tensorboard`.
  + if you want to resume a previously paused experiment you can use the `--load` flag which can continue the training from _best_ or _latest_ epoch.
  + if you want to use your model only for evaluation on the test set, add the `--test` flag.
 
## Output
For each experiment `{name}` a folder with the same name is created in the folder `root-dir/{name}/runs`
 This folder contains the following items:

```bash
.
├── best model (\*.pth.tar) # the currently best model for the experiment is saved
├── latest model (\*.pth.tar) # the currently latest model for the experiment is saved
├── config.json  # experiment hyperparameters
├── log.text  # outputs in the terminal
├── trlog.pth  # scores and metrics from all training epochs (loss, learning rate, accuracy,etc.)
├── tensorboard  # experiment values saved in tensorboard format
 ```

### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately: `tensorboard --logdir /root-dir/{name}/runs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006)
If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard.
  
  
 ## Requirements
 - Python 3.7
 - Pytorch
 - Tensorboard 1.14

## References

```bash
@inproceedings{maml,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International conference on machine learning},
  pages={1126--1135},
  year={2017},
  organization={PMLR}
}

@MISC{dataflowr,
    HOWPUBLISHED = "\url{https://mlelarge.github.io/dataflowr-web/dldiy.html}",
    TITLE = "Ecole polytechnique, MAP 583 — Deep Learning Deep Learning : Do-It-Yourself ! — dataflowr",
    AUTHOR = "Marc LeLarge and Andrei Bursuc"
}

@article{matchingnetworks,
  title={Matching networks for one shot learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Wierstra, Daan and others},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}

@article{prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@misc{pytorch_maml,
  title={maml in pytorch - re-implementation and beyond},
  author={Mu, Fangzhou},
  howpublished={\url{https://github.com/fmu2/PyTorch-MAML}},
  year={2020}
}

@misc{MAML_Pytorch,
  author = {Liangqu Long},
  title = {MAML-Pytorch Implementation},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
  commit = {master}
}

@misc{maml-authors,
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  title = {Model-Agnostic Meta-Learning},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cbfinn/maml}}
}


@article{Arnold2020-ss,
  title         = "learn2learn: A Library for {Meta-Learning} Research",
  author        = "Arnold, S{\'e}bastien M R and Mahajan, Praateek and Datta,
                   Debajyoti and Bunner, Ian and Zarkias, Konstantinos Saitas",
  month         =  aug,
  year          =  2020,
  url           = "http://arxiv.org/abs/2008.12284",
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "2008.12284"
}
