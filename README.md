# Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning

Hello - Welcome! :beers:

This is the official repository for the paper [Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning](https://arxiv.org/abs/2304.03916) (ICML 2023).

## Install Prerequisites
Select and install the correct version of [PyTorch](https://pytorch.org/get-started/previous-versions/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries), and also install [CLIP](https://github.com/openai/CLIP) and [WILDS](https://github.com/p-lambda/wilds). 

Example commands for installation on Linux with CUDA>=11.1:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyg pytorch-scatter -c pyg
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install wilds
pip install opencv-python
```

Install Matplotlib and Plotly for visualization:
```
conda install -c conda-forge matplotlib
conda install -c plotly plotly=5.9.0
conda install -c conda-forge python-kaleido
```

(Optional) Install Wandb for visualization:
```
conda install -c conda-forge wandb
```


## Prepare the Datasets
### Waterbirds Segmentations
```
cd data
wget https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz
tar -xf segmentations.tgz -C ./waterbirds
```

### ImageNet-1K
```
conda install -c huggingface -c conda-forge datasets
cd data
wget https://image-net.org/data/bboxes_annotations.tar.gz
mkdir ./imagenet/bboxes
tar -xf bboxes_annotations.tar.gz -C ./imagenet/bboxes
```

## Run Experiments

Launch spurious detection
```
python tools/detection_command_launchers.py --device 0 1 3 4 --class_rank_range 0 100 --output_dir detection_runs
```

Evaluate the pre-trained models
```
python run_expt.py --dataset waterbirds --algorithm ERM --model clip-vit --root_dir /media/SSD2/Dataset --device 2 --seed 11111111 --use_wandb --eval_only --eval_split test --eval_epoch -1
```
Train ERM

 CUDA_VISIBLE_DEVICES=2 nohup python run_expt.py  --dataset waterbirds --algorithm ERM --finetuning linear --model clip-rn50 --root_dir /media/SSD2/Dataset --device 0 --freeze_language --freeze_vision --seed 11111111 --batch_size 128 --n_epochs 200 --weight_decay 1e-4 --lr 1e-4 --scheduler cosine_schedule_with_warmup --uniform_over_groups --use_wandb --download=True > nohup_linear.out &

 CUDA_VISIBLE_DEVICES=2 nohup python run_expt.py  --dataset waterbirds --algorithm ERM --model clip-rn50 --root_dir /media/SSD2/Dataset --device 0 --freeze_language --freeze_vision --train_projection --seed 11111111 --batch_size 128 --n_epochs 200 --weight_decay 1e-4 --lr 1e-4 --scheduler cosine_schedule_with_warmup --uniform_over_groups --use_wandb --download=True > nohup_linear.out &


Train the models
```
python run_expt.py  --dataset waterbirds --algorithm Multimodal --model clip-rn50 --root_dir /media/SSD2/Dataset --device 0 --freeze_language --freeze_vision --train_projection --seed 11111111 --batch_size 128 --n_epochs 300 --class_weight 0 --clip_weight 1.0 --image_weight 1.0 --language_weight 1.0 --domain_weight 0.0 --spurious_weight 0.0 --spurious_class_weight 1.0 --spurious_clip_weight 0.0 --crossmodal_weight 0.0 --pos_weight 1.0 --neg_weight 1.0 --weight_decay 1e-5 --lr 1e-4 --use_wandb --download=True
```
python run_expt.py  --dataset waterbirds --algorithm Multimodal --model clip-vit --root_dir /media/SSD2/Dataset --device 0 --freeze_language --freeze_vision --train_projection --seed 11111111 --batch_size 128 --n_epochs 300 --class_weight 0 --clip_weight 1.0 --image_weight 0.0 --language_weight 0.0 --domain_weight 0.0 --spurious_weight 0.0 --spurious_class_weight 0.0 --spurious_clip_weight 0.0 --crossmodal_weight 0.0 --pos_weight 0.0 --neg_weight 0.0 --weight_decay 1e-5 --lr 1e-4 --use_wandb --download=True

CUDA_VISIBLE_DEVICES=2 nohup python run_expt.py  --dataset waterbirds --algorithm Multimodal --model clip-rn50 --root_dir /media/SSD2/Dataset --device 0 --freeze_language --freeze_vision --train_projection --seed 11111111 --batch_size 128 --n_epochs 200 --class_weight 0 --clip_weight 1.0 --image_weight 0.0 --language_weight 0.0 --domain_weight 0.0 --spurious_weight 0.0 --spurious_class_weight 0.0 --spurious_clip_weight 0.0 --crossmodal_weight 0.0 --pos_weight 0.0 --neg_weight 0.0 --weight_decay 1e-4 --lr 1e-4 --scheduler cosine_schedule_with_warmup --uniform_over_groups --use_wandb --download=True > nohup_linear_clip.out &

## References
* [WILDS](https://github.com/p-lambda/wilds)
* [CLIP](https://github.com/openai/CLIP)


## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{yu2023mitigating,
  title={Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning},
  author={Yang, Yu and Nushi, Besmira and Palangi, Hamid and Mirzasoleiman, Baharan},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```