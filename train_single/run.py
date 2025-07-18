#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from train_single.options import get_options
from train_single.train import train_epoch, validate, get_inner_model
from train_single.reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem


def run(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        if opts.problem == 'utpp':
            tb_logger = TbLogger(os.path.join(opts.log_dir,
                                              '{}_{}_{}'.format(opts.problem, opts.graph_size, opts.num_products),
                                              opts.run_name))
        elif opts.problem == 'rtpp':
            tb_logger = TbLogger(os.path.join(opts.log_dir,
                                              '{}_{}_{}_{}'.format(opts.problem, opts.graph_size, opts.num_products, str(opts.coeff)),
                                              opts.run_name))
        else:
            tb_logger = TbLogger(os.path.join(opts.log_dir,
                                              '{}_{}_{}'.format(opts.problem, opts.graph_size, opts.num_products),
                                              opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device('cuda:0' if opts.use_cuda else 'cpu')

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    load_path = opts.load_path if opts.load_path is not None else opts.resume_path
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel
    }.get(opts.model, None)
    assert model_class is not None, 'Unknown model: {}'.format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_products=opts.num_products,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, 'Unknown baseline: {}'.format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    if problem.NAME == 'utpp':
        val_dataset = problem.make_dataset(
            filename=opts.val_dataset, num_samples=opts.val_size,
            graph_size=opts.graph_size, num_products=opts.num_products, max_price=opts.max_price)
    elif problem.NAME == 'rtpp':
        val_dataset = problem.make_dataset(
            filename=opts.val_dataset, num_samples=opts.val_size,
            graph_size=opts.graph_size, num_products=opts.num_products, max_price=opts.max_price,
            max_supply=opts.max_supply, coeff=opts.coeff)

    if opts.resume_path:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume_path)[-1])[0].split('-')[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print('Resuming after {}'.format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

if __name__ == '__main__':
    run(get_options())
