# Experiments Check List

## Training 5 way 1 shot classification ConvNet4 (batch size = 100, episodes = 4, epoch = 50)
```python3 run.py --root-dir /content/drive/MyDrive/data --config /content/few_shot_classification/configs/convnet4/mini_imagenet/5_way_1_shot/reproduce_maml.yaml --name 5_way_1_shot_batch100_epoch50 --gpu=0,1 --meta --tensorboard --val```
```python3 run.py --root-dir /content/drive/MyDrive/data --config /content/few_shot_classification/configs/convnet4/mini_imagenet/5_way_1_shot/reproduce_maml.yaml --name 5_way_1_shot_batch100_epoch50 --gpu=0,1 --meta --tensorboard --load best --test```
