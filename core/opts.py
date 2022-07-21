from argparse import Namespace

trainer_opts = {
    'epoch_num': 500,
    'device': 'cuda:0',
    'ckpt_start': 'finetuned_head.pt'
}

optimization_opts = {
    'batch_size': 512,
    'epoch_num': 500,
    'lr': 0.001,
    'betas': [0.9, 0.999],
    'weight_decay': 5e-4,
    'max_lr': 0.001
}

dataset_opts = {
    'scale': 4,
    'original_size': 10496,
}

optimization_opts = Namespace(**optimization_opts)
dataset_opts = Namespace(**dataset_opts)
trainer_opts = Namespace(**trainer_opts)