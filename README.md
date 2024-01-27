<h1 align='center'>DiT-Pytorch</h1>

## [Scalable Diffusion Models with Transformers (DiT)](http://arxiv.org/abs/2212.09748)

## Precautions
This code mainly comes from facebookresearch [official code](https://github.com/facebookresearch/DiT/). Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_path___, ___global-batch-size___ and ___num-classes___ parameters. 

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___.
```
# opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
opt = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0)
```

## Train this model

### Note:
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1___ on each GPU, otherwise BN layer maybe report an error. If you recive an error like "error: unrecognized arguments: --local-rank=1" when you use distributed multi-GPUs training, just replace the command ___"torch.distributed.launch"___ to ___"torch.distributed.run"___.

### train model with single-machine single-card：
```
python train_gpu.py
```

### train model with single-machine multi-card：
```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-card: 
(using a specified part of the cards: for example, I want to use the second and fourth cards)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-card:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain card, just add CUDA_VISIBLE_DEVICES= to specify the index number of the card before each command. The principle is the same as single-machine multi-card training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```
