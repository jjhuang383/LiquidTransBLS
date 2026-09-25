"""Train and predict with the full LiquidTransBLS model only."""
import argparse
import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src import LiquidTransBLS
from src.data_loader import (
    CATLLoader, CMAPSSLoader, IMSLoader, NCMAPSSLoader, PHM2012Loader,
    TJULoader, XJTULoader, configure_bearing_experiment,
)
from src.preprocessing import (
    BEARING_DATASETS, TaskSpecificScaler, apply_odometer,
    create_sliding_window, postprocess_predictions, sample_coreset,
)


DATASETS = {
    'CMAPSS': (CMAPSSLoader, 'C-MAPSS', 'CMAPSS', 80),
    'PHM2012': (PHM2012Loader, 'PHM 2012', 'PHM 2012', 20),
    'XJTU': (XJTULoader, 'XJTU', 'XJTU', 10),
    'IMS': (IMSLoader, 'IMS', 'IMS', 30),
    'N-CMAPSS': (NCMAPSSLoader, 'N-CMAPSS', 'N-CMAPSS', 80),
    'CATL': (CATLLoader, 'CATL', 'CATL', 80),
    'TJU': (TJULoader, 'TJU', 'TJU', 30),
}


class SyntheticLoader:
    """In-memory example covering initial, same-condition and new-condition tasks."""
    tasks = {'A': 'initial', 'B': 'same_condition', 'C': 'new_condition'}

    def __init__(self, seed, length):
        self.seed = seed
        self.length = length

    def get_task_update_type(self, task):
        return self.tasks[task]

    def get_task_scaler_group(self, task):
        return 'condition_2' if task == 'C' else 'condition_1'

    def get_task_data(self, task):
        def sequence(offset):
            rng = np.random.default_rng(self.seed + offset)
            t = np.linspace(0, 1, self.length, dtype=np.float32)
            x = np.column_stack([t, t ** 2, np.exp(t), np.sin(t)])
            x += rng.normal(0, 0.03, x.shape)
            if task == 'C':
                x = 1.2 * x + 0.1
            return [x.astype(np.float32)], [(1 - t).reshape(-1, 1)]
        train = sequence(10 * ord(task))
        test = sequence(10 * ord(task) + 1)
        if task == 'B':
            return (None, None), train, test
        return train, (None, None), test


def prepare_windows(x, y, window_size):
    """Keep unit boundaries; each label belongs to the window's last row."""
    if x is None:
        return None, None
    units = x if isinstance(x, list) else [x]
    targets = y if isinstance(y, list) else [y]
    if len(units) != len(targets):
        raise ValueError('Each unit must have its own target sequence.')
    windows, labels = [], []
    for unit, target in zip(units, targets):
        if len(unit) != len(target):
            raise ValueError('Feature and target sequence lengths must match.')
        if len(unit) < window_size:
            continue
        window = create_sliding_window(unit, window_size)
        label = np.asarray(target, dtype=np.float32).reshape(-1, 1)[window_size - 1:]
        if not np.isfinite(label).all():
            raise ValueError('Targets must be finite.')
        windows.append(window.astype(np.float32))
        labels.append(label)
    return (windows, np.vstack(labels)) if windows else (None, None)


def train_task(model, source, increment, update_type, initial, epochs, lambda_phy):
    """Route updates without using the held-out prediction inputs or labels."""
    source_x, source_y = source
    inc_x, inc_y = increment
    if source_x is None and (initial or update_type == 'new_condition'):
        source_x, source_y = inc_x, inc_y
    if initial:
        if source_x is None:
            raise ValueError('Initial task has no training windows.')
        model.adapt(source_x, source_y, X_target=None, epochs=epochs,
                    lambda_mmd=0.0, lambda_phy=lambda_phy)
        model.fit(source_x, source_y)
        return 'initial'
    if update_type == 'new_condition':
        if source_x is None:
            raise ValueError('New condition has no training windows.')
        source_x, source_y = sample_coreset(source_x, source_y)
        replay_x, replay_y = model.replay_buffer.sample(model.replay_buffer.capacity)
        replay = (replay_x, replay_y) if replay_x is not None else None
        model.adapt(source_x, source_y, X_target=None, epochs=epochs,
                    lambda_mmd=0.0, lambda_phy=lambda_phy,
                    replay_data=replay, lambda_replay=1.0)
        memory_x, memory_y = model.replay_buffer.get_data()
        if memory_x is not None:
            source_x = list(source_x) + [(memory_x, False)]
            source_y = np.vstack([source_y, memory_y])
        model.replay_buffer.clear()
        model.fit(source_x, source_y)
        return 'deep'
    if inc_x is not None:
        inc_x, inc_y = sample_coreset(inc_x, inc_y)
        model.update_data(inc_x, inc_y)
        return 'head'
    return 'evaluation_only'


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    configure_bearing_experiment(label_mode='linear')
    if args.dataset == 'Synthetic':
        window_size = args.window_size or 8
        loader = SyntheticLoader(args.seed, max(64, 2 * window_size))
        dataset_name = 'Synthetic'
    else:
        loader_class, dataset_name, folder, default_window = DATASETS[args.dataset]
        root = args.data_root / folder
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset directory not found: {root}')
        loader = loader_class(data_root=str(root))
        window_size = args.window_size or default_window
    tasks = list(loader.tasks)
    if args.tasks:
        requested = [task.strip().upper() for task in args.tasks.split(',')]
        if any(task not in tasks for task in requested):
            raise ValueError(f'Available tasks: {", ".join(tasks)}')
        tasks = [task for task in tasks if task in requested]
    if not tasks:
        raise ValueError('Select at least one task.')

    scaler = TaskSpecificScaler(dataset_name)
    model = None
    bearing = dataset_name in BEARING_DATASETS
    lambda_phy = 0.0 if dataset_name in {'CATL', 'TJU'} else args.lambda_phy
    outputs = []
    for index, task in enumerate(tasks):
        update_type = loader.get_task_update_type(task)
        initial = model is None or update_type == 'initial'
        if initial:
            scaler = TaskSpecificScaler(dataset_name)
        train, inc, test = loader.get_task_data(task)
        x_train, y_train = train
        x_inc, y_inc = inc
        x_test, y_test = test
        if x_train is None and x_inc is None:
            raise ValueError(f'Task {task}: no training data; check the local dataset layout.')
        if x_test is None:
            raise ValueError(f'Task {task}: no prediction data; check the local dataset layout.')
        if bearing:
            x_train = apply_odometer(x_train, dataset_name)
            x_inc = apply_odometer(x_inc, dataset_name)
            x_test = apply_odometer(x_test, dataset_name)
        group = loader.get_task_scaler_group(task)
        if group not in scaler.group_scalers:
            scaler.fit_task(index, x_train if x_train is not None else x_inc)
            scaler.group_scalers[group] = scaler.task_scalers[index]
        scaler.task_scalers[index] = scaler.group_scalers[group]
        train = prepare_windows(scaler.transform_task(index, x_train), y_train, window_size)
        inc = prepare_windows(scaler.transform_task(index, x_inc), y_inc, window_size)
        test_x, test_y = prepare_windows(scaler.transform_task(index, x_test), y_test, window_size)
        if test_x is None:
            raise ValueError(f'Task {task}: prediction sequences are shorter than the window.')
        source = train[0] if train[0] is not None else inc[0]
        if source is None:
            raise ValueError(f'Task {task}: training sequences are shorter than the window.')
        if initial:
            config = {}
            if dataset_name == 'XJTU':
                config.update(lambda_global=0.10, lambda_stage=0.20)
            default_nodes = 4000 if dataset_name == 'PHM 2012' else 3000
            default_reg = {'PHM 2012': 1.2297288957910198e-05, 'XJTU': 1e-4}.get(dataset_name, 0.1)
            model = LiquidTransBLS(
                window_size=window_size, input_dim=source[0].shape[1] // window_size,
                d_model=args.d_model, n_heads=4, n_layers=2, dropout=0.05,
                n_enhancement_nodes=args.enhancement_nodes or default_nodes,
                reg_param=args.reg_param if args.reg_param is not None else default_reg,
                replay_capacity=args.buffer_size, is_bearing_dataset=bearing, **config,
            )
        route = train_task(model, train, inc, update_type,
                           initial, args.epochs, lambda_phy)
        predictions, raw_predictions = [], []
        unit_ids, steps = [], []
        for unit_index, windows in enumerate(test_x):
            raw = model.predict([windows]).reshape(-1)
            if not np.isfinite(raw).all():
                raise FloatingPointError(f'Task {task}: non-finite model predictions.')
            processed, _ = postprocess_predictions(raw, dataset_name)
            predictions.extend(processed.reshape(-1))
            raw_predictions.extend(raw)
            unit_ids.extend([unit_index] * len(windows))
            steps.extend(range(window_size - 1, window_size - 1 + len(windows)))
        output = pd.DataFrame({'task': task, 'unit_index': unit_ids, 'step': steps,
                               'y_true': test_y.reshape(-1), 'y_pred': predictions,
                               'y_pred_raw': raw_predictions})
        outputs.append(output)
        rmse = np.sqrt(np.mean((output.y_true - output.y_pred) ** 2))
        logging.info('Task %s | route=%s | windows=%d | normalized RMSE=%.6f',
                     task, route, len(output), rmse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['Synthetic', *DATASETS], default='Synthetic')
    parser.add_argument('--data-root', type=Path, default=Path('data'), help='Parent of dataset folders')
    parser.add_argument('--tasks', help='Task subset in predefined order, e.g. A,B,C')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window-size', type=int)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--enhancement-nodes', type=int)
    parser.add_argument('--reg-param', type=float)
    parser.add_argument('--buffer-size', type=int, default=2000)
    parser.add_argument('--lambda-phy', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    args = parser.parse_args()
    if args.epochs < 1 or args.buffer_size < 3 or args.d_model < 4 or args.d_model % 4:
        parser.error('epochs >= 1, buffer-size >= 3, and d-model >= 4 divisible by 4 are required.')
    if args.window_size is not None and args.window_size < 1:
        parser.error('window-size must be positive.')
    if args.enhancement_nodes is not None and args.enhancement_nodes < 1:
        parser.error('enhancement-nodes must be positive.')
    if args.reg_param is not None and args.reg_param <= 0:
        parser.error('reg-param must be positive for incremental updates.')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run(args)


if __name__ == '__main__':
    main()
