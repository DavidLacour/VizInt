# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m/
# --------------------------------------------------------

from typing import Optional, Union
import io
import os
import ast
import json
from pathlib import Path
from hydra.utils import instantiate
from safetensors.torch import load as load_st
from safetensors.torch import save_file

import torch

from .dist import save_on_main, is_main_process


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def load_state_dict(model, state_dict, prefix='', ignore_missing=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(
        args, iteration, model, model_without_ddp, optimizer, loss_scaler, loss_balancer=None, 
        ckpt_name=None, all_nodes=False, save_as_safetensors=False, model_args=None
    ):
    output_dir = Path(args.output_dir)
    iteration_name = str(iteration)
    ckpt_name = ckpt_name or iteration_name

    # Only create the save_dict on the main process, unless all_nodes is set to True
    if is_main_process() or (all_nodes and args.gpu == 0): 
        checkpoint_path = os.path.join(output_dir, f'checkpoint-{ckpt_name}.pth')

        to_save = {
            'model': model_without_ddp.state_dict(),
            'iteration': iteration,
            'args': args,
            'scaler': loss_scaler.state_dict(),
        }

        if optimizer is not None:
            to_save['optimizer'] = optimizer.state_dict()

        if loss_balancer is not None:
            to_save['loss_balancer'] = loss_balancer.state_dict()

        save_on_main(to_save, checkpoint_path)

        # Save only weights as .safetensors, including model args as metadata
        if save_as_safetensors:
            checkpoint_path_st = os.path.join(output_dir, f"checkpoint-{ckpt_name}.safetensors")
            save_safetensors(to_save["model"], checkpoint_path_st, metadata_dict=model_args)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    # torch.amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            
    if args.resume:
        print("Auto resume checkpoint: %s" % args.resume)

        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu')
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'iteration' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_iteration = checkpoint['iteration'] + 1

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def save_safetensors(state_dict, ckpt_path, metadata_dict=None):
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
    if metadata_dict is not None:
        metadata = {k: str(v) for k, v in metadata_dict.items()}
    else:
        metadata = None
    save_file(state_dict, ckpt_path, metadata=metadata)


def parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        try:
            v_parsed = ast.literal_eval(v)
        except:
            v_parsed = v
        metadata[k] = v_parsed
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, "rb") as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors

    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})
    metadata = parse_metadata(metadata)

    return tensors, metadata


def load_model_from_safetensors(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    to_eval: bool = True,
) -> torch.nn.Module:
    ckpt, config = load_safetensors(ckpt_path)
    model = instantiate(config)
    model.load_state_dict(ckpt)
    if device is not None:
        model = model.to(device)
    if to_eval:
        model = model.eval()
    return model

# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on DETR, MMCV, and 4M code bases
# https://github.com/facebookresearch/detr
# https://github.com/open-mmlab/mmcv
# https://github.com/apple/ml-4m/
# --------------------------------------------------------

import os
import pickle
import shutil
import sys
import tempfile
import datetime

import torch
import torch.distributed as dist


def setup_for_distributed(is_main):
    """
    This function disables printing when not in main process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force or kwargs.get('file', None) == sys.stderr:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_main(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_on_all(*args, **kwargs):
    torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    # Set timeout to 1h20 in case some long download of dataset has to happen
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, 
                                         timeout=datetime.timedelta(4800))
    torch.distributed.barrier()
    if ("print_all" not in args) or (not args.print_all):
        setup_for_distributed(args.rank == 0)

# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on DETR and 4M code bases
# https://github.com/facebookresearch/detr
# https://github.com/apple/ml-4m/
# --------------------------------------------------------

import datetime
import logging
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

try:
    import wandb
except:
    pass

from .dist import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, iter_len=None, header=None, start_iter=0):
        iter_len = iter_len if iter_len is not None else len(iterable)
        i = start_iter or 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(iter_len))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == iter_len - 1:
                if iter_len > 0:
                    eta_seconds = iter_time.global_avg * (iter_len - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = "?"
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            iter_len if iter_len > 0 else "?",
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            iter_len if iter_len > 0 else "?",
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        time_per_iter_str = "{:.4f}".format(total_time / iter_len) if iter_len > 0 else "?"
        print("{} Total time: {} ({} s / it)".format(header, total_time_str, time_per_iter_str))


class WandbLogger(object):
    def __init__(self, args):
        wandb.init(
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=getattr(args, "wandb_group", None),
            name=getattr(args, "wandb_run_name", None),
            tags=getattr(args, "wandb_tags", None),
            mode=getattr(args, "wandb_mode", "online"),
        )

    @staticmethod
    def safe_log(*args, **kwargs):
        try:
            wandb.log(*args, **kwargs)
        except (wandb.CommError, BrokenPipeError):
            logging.error("wandb logging failed, skipping...")

    def safe_log_image(self, image, key, **kwargs):
        # image can be a numpy array, PIL image, or matplotlib fig
        log_dict = {key: wandb.Image(image)}
        self.safe_log(log_dict, **kwargs)

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v

        self.safe_log(log_dict, step=self.step)

    def flush(self):
        pass

    def finish(self):
        try:
            wandb.finish()
        except (wandb.CommError, BrokenPipeError):
            logging.error("wandb failed to finish")


# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on:
# https://github.com/apple/ml-4m/
# --------------------------------------------------------
def setup_run_name(args):
    if args.run_name == 'auto':
        # This returns the config name after removing the first two parent dirs and extension
        args.run_name = args.config_path.partition('cfgs/')[2].replace(".yaml", "")

    if 'wandb_run_name' in args and args.wandb_run_name == 'auto':
        # Wandb omits the current parent dir (pretrain, finetune, etc...) as it is part of the wandb project
        args.wandb_run_name = args.run_name.partition('/')[2]

    if 'output_dir' in args and 'auto' in args.output_dir:
        args.output_dir = args.output_dir.replace('auto', args.run_name)



# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on:
# https://github.com/apple/ml-4m/
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np


def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0):
    # Compatible with batching
    # From https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    if top_k > 0.0:
        if isinstance(top_k, int):
            k = min(top_k, logits.shape[-1])
        elif isinstance(top_k, float):
            k = min(int(top_k * logits.shape[-1]), logits.shape[-1])
        else:
            raise ValueError(f"Invalid value for top_k: {top_k}")

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        restore_indices = torch.argsort(sorted_indices, dim=-1)
        indices_to_remove = torch.gather(sorted_indices_to_remove, dim=-1, index=restore_indices)
        logits[indices_to_remove] = float("-inf")

    return logits

def sample_tokens(logits, temperature=1.0, top_k=0.0, top_p=0.0):
    if np.isclose(temperature, 0, atol=1e-10):
        samples = torch.argmax(logits, dim=-1)
        # Since argmax is used, all sampled_probs will be 1 as we're selecting the max probability
        sampled_probs = torch.ones_like(samples, dtype=torch.float32)
    else:
        filtered_logits = top_k_top_p_filtering(logits, top_k, top_p)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        samples = torch.multinomial(probs, 1)[:, 0]
        sampled_probs = probs[torch.arange(len(samples)), samples]
    return samples, sampled_probs



# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on DINO code base
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import numpy as np
import math

def cosine_scheduler(base_value, final_value, total_iters, warmup_iters, start_warmup_value=0.0):
    assert warmup_iters >= 0
    assert total_iters > 0
    assert start_warmup_value <= base_value
    assert base_value >= final_value

    if warmup_iters > 0:
        print("Set warmup iters = %d" % warmup_iters)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        print("No warmup iters")
        warmup_schedule = np.array([])

    cosine_iters = np.arange(total_iters - warmup_iters)
    cosine_schedule = np.array([
        final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(cosine_iters)))) 
        for i in cosine_iters
    ])

    schedule = np.concatenate((warmup_schedule, cosine_schedule))

    assert len(schedule) == total_iters
    return schedule