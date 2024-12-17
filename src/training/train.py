import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import random
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
import pickle
from tqdm import tqdm
from copy import deepcopy

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
)

import collections
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
        
        
def train_one_epoch_original_clip(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    use_gt=False,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    class_names = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].class_names
    templates = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].templates

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            if args.lr_scheduler == "one_cycle":
                scheduler.step()
            else:
                scheduler(step)
        
        # if len(batch) == 4:
        # images, distill_images, texts, common_texts, text_raws, metadata, zeroshot_classnames = batch
        
        images, distill_images, texts, common_texts, text_raws, \
            label_text, per_image_texts, per_image_group_texts, \
            metadata, zeroshot_classnames = batch
        
        label_texts = []
        for i in range(len(zeroshot_classnames)):
            # if use_gt:
            gt_classname = metadata[i]['class_name'].lower()
            # else:
            zeroshot_classname = zeroshot_classnames[i][0]
            # print(zeroshot_classname, gt_classname)
            if use_gt:
                label_texts.append(templates[0](gt_classname))
            else:
                label_texts.append(templates[0](zeroshot_classname))
            
        
        label_text_tokens = model.tokenizer(label_texts).to(device=device, non_blocking=True)
        label_text_features = model.encode_text(label_text_tokens, normalize=True)
        
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)

        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # torch.autograd.set_detect_anomaly(True)
                bs, k, dim = texts.shape
               
               
                
                image_features = model.encode_image(images, normalize=True)
                logit_scale = model.logit_scale.exp()
                
                # text_features = model.encode_text(label_texts.reshape(-1, dim), normalize=True)
                # text_features = text_features.detach()
                
                losses = loss(image_features=image_features,
                              text_features=label_text_features,
                              logit_scale=logit_scale,
                            #   args=args,
                              output_dict=True)
                
            
                        
               
                total_loss = sum(losses.values())
                
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            raise NotImplemented()
            
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
       
        
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx



def compute_text_weights(text_features, prototypes, preds):
    bs, dim = text_features.shape
    weight = torch.bmm(text_features.unsqueeze(1), prototypes.T.unsqueeze(0).expand(bs, -1, -1))
    weight = weight.squeeze(1)
    # weight = weight / weight.sum(dim=1, keepdim=True)
    # jsd_score = compute_js_divergence_batch(weight)
    top2_vals, top2_indices = torch.topk(weight, 2, dim=1)

    weight = top2_vals[:, 0] - top2_vals[:, 1]
    mask = top2_indices[:, 0] == preds
    # weight = weight * mask.float()
    return weight


def train_one_epoch_v2(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    class_names = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].class_names


    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    templates = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].templates

    

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    class_dist_ours = collections.defaultdict(int)
    class_dist_zeroshot = collections.defaultdict(int)
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    prototypes = []
    for classname in class_names:
        prototypes.append(model.memory_bank[classname])
    prototypes = torch.stack(prototypes)
    
    # use_zeroshot = False
    # # zeroshot_ratio = 1.0 - epoch / args.epochs
    # # zeroshot_ratio = epoch * 1.0 / args.epochs
    # zeroshot_ratio = 0.5
    # if np.random.rand() < zeroshot_ratio:
    #     use_zeroshot = True
    for i, batch in tqdm(enumerate(dataloader), total=num_batches_per_epoch,  desc="Training"):
    # for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            if args.lr_scheduler == "one_cycle":
                scheduler.step()
            else:
                scheduler(step)
        
        # if len(batch) == 4:
        # images, distill_images, texts, common_texts, text_raws, metadata, zeroshot_classnames = batch
        
        images, distill_images, texts, common_texts, text_raws, \
            _, per_image_texts, per_image_group_texts, \
            metadata, zeroshot_classnames = batch
        
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        distill_images = distill_images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # common_texts = common_texts.to(device=device, non_blocking=True)
        per_image_group_texts = per_image_group_texts.to(device=device, non_blocking=True)
        # label_texts = label_texts.to(device=device, non_blocking=True)
        per_image_texts = per_image_texts.to(device=device, non_blocking=True)

        mem_classifier = []
        for i, classname in enumerate(class_names):
            mem_classifier.append(model.memory_bank[classname])
        mem_classifier = torch.stack(mem_classifier)
        mem_classifier = F.normalize(mem_classifier, dim=1)
        classifier = mem_classifier.T
        classname2id = {classname: i for i, classname in enumerate(class_names)}
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        
            
        if args.accum_freq == 1:
            with autocast():
                # torch.autograd.set_detect_anomaly(True)
                bs, k, dim = texts.shape
               
                
               
                image_features = model.encode_image(images, normalize=True)
                logit_scale = model.logit_scale.exp()
                 
                membank_features = []
                membank_features_zeroshot = []
                
                logits = 100.0 * image_features @ classifier
                preds = logits.argmax(dim=1)
                zeroshot_preds = torch.zeros_like(preds)
                label_texts = []
                label_texts_zeroshot = []
                for i in range(len(zeroshot_classnames)):
                    zeroshot_classname = zeroshot_classnames[i][0]
                    zeroshot_preds[i] = classname2id[zeroshot_classname]
                    
                    
                    classname = class_names[preds[i]]
                    # label_texts.append(f"a photo of a {classname}.")
                    # label_texts_zeroshot.append(f"a photo of a {zeroshot_classname}.")
                    label_texts.append(templates[0](classname))
                    label_texts_zeroshot.append(templates[0](zeroshot_classname))
                  
                    # class_dist_ours[classname] += 1
                    # class_dist_zeroshot[zeroshot_classname] += 1
                    membank_features.append(model.memory_bank[classname])
                    membank_features_zeroshot.append(model.memory_bank[zeroshot_classname])
                membank_features = torch.stack(membank_features)
                membank_features_zeroshot = torch.stack(membank_features_zeroshot)
                
                label_text_tokens = model.tokenizer(label_texts).to(device=device, non_blocking=True)
                label_text_tokens_zeroshot = model.tokenizer(label_texts_zeroshot).to(device=device, non_blocking=True)
                
                
                label_text_features = model.encode_text(label_text_tokens, normalize=True)
                label_text_features_zeroshot = model.encode_text(label_text_tokens_zeroshot, normalize=True)
                
                # label_text_features = model.encode_text(label_texts.reshape(-1, dim), normalize=True)
                per_image_text_features = model.encode_text(per_image_texts.reshape(-1, dim), normalize=True)
                per_image_group_text_features = model.encode_text(per_image_group_texts.reshape(-1, dim), normalize=True)
                
                per_image_text_weights = compute_text_weights(per_image_text_features, prototypes, preds).detach() + 1e-6
                per_image_group_text_weights = compute_text_weights(per_image_group_text_features, prototypes, preds).detach() + 1e-6
                per_image_text_weights_zeroshot = compute_text_weights(per_image_text_features, prototypes, zeroshot_preds).detach() + 1e-6
                per_image_group_text_weights_zeroshot = compute_text_weights(per_image_group_text_features, prototypes, zeroshot_preds).detach() + 1e-6
                label_text_weight = compute_text_weights(label_text_features, prototypes, preds).detach() + 1e-6
                label_text_weight_zeroshot = compute_text_weights(label_text_features_zeroshot, prototypes, zeroshot_preds).detach() + 1e-6
                
                # label_text_weight = per_image_text_weights + per_image_group_text_weights
                # label_text_weight_zeroshot = per_image_text_weights_zeroshot + per_image_group_text_weights_zeroshot
                # per_image_text_weights = torch.ones_like(per_image_text_weights)
                # per_image_group_text_weights = torch.ones_like(per_image_group_text_weights)
                # per_image_text_weights_zeroshot = torch.ones_like(per_image_text_weights_zeroshot)
                # per_image_group_text_weights_zeroshot = torch.ones_like(per_image_group_text_weights_zeroshot)
                # label_text_weight = torch.ones_like(label_text_weight)
                # label_text_weight_zeroshot = torch.ones_like(label_text_weight_zeroshot)
                
                per_image_group_text_weights = per_image_group_text_weights * args.use_batch_caption
                per_image_group_text_weights_zeroshot = per_image_group_text_weights_zeroshot * args.use_batch_caption
                
                per_image_text_weights = per_image_text_weights * args.use_image_caption
                per_image_text_weights_zeroshot = per_image_text_weights_zeroshot * args.use_image_caption
                
                # import pdb;pdb.set_trace()
                # label_text_weight = 1.0 * args.use_template_caption
                label_text_weight = label_text_weight * args.use_template_caption
                label_text_weight_zeroshot = label_text_weight_zeroshot * args.use_template_caption
                
            
                total_weight = label_text_weight + per_image_text_weights + per_image_group_text_weights
                total_weight_zeroshot = label_text_weight_zeroshot + per_image_text_weights_zeroshot + per_image_group_text_weights_zeroshot
              
         
                text_features = label_text_weight * label_text_features + \
                    per_image_text_features * per_image_text_weights.unsqueeze(1) + \
                    per_image_group_text_features * per_image_group_text_weights.unsqueeze(1)
                text_features = text_features / total_weight.unsqueeze(1)
                
                text_features_zeroshot = label_text_weight * label_text_features_zeroshot + \
                    per_image_text_features * per_image_text_weights_zeroshot.unsqueeze(1) + \
                    per_image_group_text_features * per_image_group_text_weights_zeroshot.unsqueeze(1)
                text_features_zeroshot = text_features_zeroshot / total_weight_zeroshot.unsqueeze(1)
               
                # Get prototype for each image based on its pseudo-label
                text_features = membank_features +  args.alpha * (text_features - membank_features) 
                text_features_zeroshot = membank_features_zeroshot +  args.alpha * (text_features_zeroshot - membank_features_zeroshot)
                
                
                losses = loss(image_features=image_features,
                              text_features=text_features,
                              logit_scale=logit_scale,
                              output_dict=True) 

                losses_zeroshot = loss(image_features=image_features,
                                text_features=text_features_zeroshot,
                                logit_scale=logit_scale,
                                output_dict=True) 
                
                losses["zeroshot"] = sum(losses_zeroshot.values()) * args.use_zeroshot_pseudolabel
                total_loss = sum(losses.values()) * args.use_finetune_pseudolabel
                
                losses["loss"] = total_loss

            backward(total_loss, scaler)
            
            with torch.no_grad():
                temp_bank = {}
                cnt = collections.defaultdict(int)
                for i in range(len(zeroshot_classnames)):
                    proto_classname = class_names[preds[i]]
                    zeroshot_classname = class_names[zeroshot_preds[i]]
                    # zeroshot_classname = zeroshot_classnames[i][0]
                    if zeroshot_classname not in temp_bank:
                        temp_bank[zeroshot_classname] = torch.zeros_like(text_features[i])
                    if proto_classname not in temp_bank:
                        temp_bank[proto_classname] = torch.zeros_like(text_features[i])
                    # Update the prototype with text features
                    # text_image_features = text_features[i] * (1-args.gamma) + image_features[i] * args.gamma
                    

                    temp_bank[zeroshot_classname] += text_features_zeroshot[i]
                    temp_bank[proto_classname] += text_features[i]
                    cnt[zeroshot_classname] += 1
                    cnt[proto_classname] += 1
                    
                for classname in temp_bank:
                    model.memory_bank[classname] = temp_bank[classname] / cnt[classname]
                    model.memory_bank[classname] = F.normalize(model.memory_bank[classname], dim=0)
        else:
            raise NotImplemented()
            
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
       
        
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    print("ours", class_dist_ours)
    print("zeroshot", class_dist_zeroshot)
    # end for


def extract_group_weights(
    model,
    data,
    epoch,
    args,
    tokenizer
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    class_names = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].class_names


    model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    templates = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].templates

    

    data_time_m = AverageMeter()
    end = time.time()
    
    prototypes = []
    for classname in class_names:
        prototypes.append(model.memory_bank[classname])
    prototypes = torch.stack(prototypes)
    
    # use_zeroshot = False
    # # zeroshot_ratio = 1.0 - epoch / args.epochs
    # # zeroshot_ratio = epoch * 1.0 / args.epochs
    # zeroshot_ratio = 0.5
    # if np.random.rand() < zeroshot_ratio:
    #     use_zeroshot = True
    group_weights = []
    items = []
    for i, batch in tqdm(enumerate(dataloader), total=num_batches_per_epoch,  desc="Training"):
        
        # if len(batch) == 4:
        # images, distill_images, texts, common_texts, text_raws, metadata, zeroshot_classnames = batch
        
        images, distill_images, texts, common_texts, text_raws, \
            _, per_image_texts, per_image_group_texts, \
            metadata, zeroshot_classnames = batch
        
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        distill_images = distill_images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # common_texts = common_texts.to(device=device, non_blocking=True)
        per_image_group_texts = per_image_group_texts.to(device=device, non_blocking=True)
        # label_texts = label_texts.to(device=device, non_blocking=True)
        per_image_texts = per_image_texts.to(device=device, non_blocking=True)

        mem_classifier = []
        for i, classname in enumerate(class_names):
            mem_classifier.append(model.memory_bank[classname])
        mem_classifier = torch.stack(mem_classifier)
        mem_classifier = F.normalize(mem_classifier, dim=1)
        classifier = mem_classifier.T
        classname2id = {classname: i for i, classname in enumerate(class_names)}
        
        data_time_m.update(time.time() - end)
      

        
            
        if args.accum_freq == 1:
            with autocast():
                # torch.autograd.set_detect_anomaly(True)
                bs, k, dim = texts.shape
               
                
               
                image_features = model.encode_image(images, normalize=True)
                logit_scale = model.logit_scale.exp()
                 
                membank_features = []
                membank_features_zeroshot = []
                
                logits = 100.0 * image_features @ classifier
                preds = logits.argmax(dim=1)
                zeroshot_preds = torch.zeros_like(preds)
                label_texts = []
                label_texts_zeroshot = []
                for i in range(len(zeroshot_classnames)):
                    zeroshot_classname = zeroshot_classnames[i][0]
                    zeroshot_preds[i] = classname2id[zeroshot_classname]
                    
                    
                    classname = class_names[preds[i]]
                    # label_texts.append(f"a photo of a {classname}.")
                    # label_texts_zeroshot.append(f"a photo of a {zeroshot_classname}.")
                    label_texts.append(templates[0](classname))
                    label_texts_zeroshot.append(templates[0](zeroshot_classname))
                    
                    # zs_lb.append(zeroshot_classname)
                    # ft_lb.append(classname)
                    # gt_lb.append(metadata[i]['class_name'])
                    # images_ids.append(metadata[i]['image_id'])
                    
                    item = {
                        'zs_lb': zeroshot_classname,
                        'ft_lb': classname,
                        'gt_lb': metadata[i]['class_name'],
                        'image_id': metadata[i]['image_id'],
                        "per_image_text": text_raws[i][0],
                        "per_image_group_text": common_texts[i][0]
                    }
                    items.append(item)
                    # import pdb;pdb.set_trace()
                    
                    # class_dist_ours[classname] += 1
                    # class_dist_zeroshot[zeroshot_classname] += 1
                    membank_features.append(model.memory_bank[classname])
                    membank_features_zeroshot.append(model.memory_bank[zeroshot_classname])
                membank_features = torch.stack(membank_features)
                membank_features_zeroshot = torch.stack(membank_features_zeroshot)
                
                label_text_tokens = model.tokenizer(label_texts).to(device=device, non_blocking=True)
                label_text_tokens_zeroshot = model.tokenizer(label_texts_zeroshot).to(device=device, non_blocking=True)
                
                
                label_text_features = model.encode_text(label_text_tokens, normalize=True)
                label_text_features_zeroshot = model.encode_text(label_text_tokens_zeroshot, normalize=True)
                
                # label_text_features = model.encode_text(label_texts.reshape(-1, dim), normalize=True)
                per_image_text_features = model.encode_text(per_image_texts.reshape(-1, dim), normalize=True)
                per_image_group_text_features = model.encode_text(per_image_group_texts.reshape(-1, dim), normalize=True)
                
                per_image_text_weights = compute_text_weights(per_image_text_features, prototypes, preds).detach() + 1e-6
                per_image_group_text_weights = compute_text_weights(per_image_group_text_features, prototypes, preds).detach() + 1e-6
                per_image_text_weights_zeroshot = compute_text_weights(per_image_text_features, prototypes, zeroshot_preds).detach() + 1e-6
                per_image_group_text_weights_zeroshot = compute_text_weights(per_image_group_text_features, prototypes, zeroshot_preds).detach() + 1e-6
                label_text_weight = compute_text_weights(label_text_features, prototypes, preds).detach() + 1e-6
                label_text_weight_zeroshot = compute_text_weights(label_text_features_zeroshot, prototypes, zeroshot_preds).detach() + 1e-6
                
                total_weight = label_text_weight + per_image_text_weights + per_image_group_text_weights
                total_weight_zeroshot = label_text_weight_zeroshot + per_image_text_weights_zeroshot + per_image_group_text_weights_zeroshot
                group_weight = per_image_group_text_weights / total_weight
                group_weights.append(group_weight.detach().cpu().numpy())
                
    group_weights = np.concatenate(group_weights, axis=0)
    save_path = os.path.join(args.extract_group_weight_path, f"group_weights.npy")
    # save_path = os.path.join(args.extract_group_weight_path, f"group_weights_4corrects.npy")
    np.save(save_path, group_weights)
    print(f"Group weights saved at {save_path}")
    
    import json

    json_save_path = os.path.join(args.extract_group_weight_path, f"all_labels.json")
    # Assuming all lists have the same length
    # result = []
    # for zs, ft, gt, img_id in zip(zs_lb, ft_lb, gt_lb, images_ids):
    #     item = {
    #         "zs_lb": zs,
    #         "ft_lb": ft,
    #         "gt_lb": gt,
    #         "image_id": img_id
    #     }
    #     result.append(item)

    # Save to JSON file
    with open(json_save_path, 'w') as f:
        json.dump(items, f, indent=2)
        print(f"Labels saved at {json_save_path}")
    

def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            if args.lr_scheduler == "one_cycle":
                scheduler.step()
            else:
                scheduler(step)
        
        # if len(batch) == 4:
        images, distill_images, texts, common_texts, text_raws, \
            label_texts, per_image_texts, per_image_group_texts, \
            metadata, zeroshot_classnames = batch
        
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        distill_images = distill_images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        common_texts = common_texts.to(device=device, non_blocking=True)
        label_texts = label_texts.to(device=device, non_blocking=True)
        per_image_texts = per_image_texts.to(device=device, non_blocking=True)
        per_image_group_texts = per_image_group_texts.to(device=device, non_blocking=True)
        
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # torch.autograd.set_detect_anomaly(True)
                bs, k, dim = texts.shape
               
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(distill_images, texts.reshape(-1, dim))
                    model_out.update(
                        {f"dist_{k}": v for k, v in dist_model_out.items()}
                    )
                
                image_features = model.encode_image(images, normalize=False)
                logit_scale = model.logit_scale.exp()
                
               
                
                label_text_features = model.encode_text(label_texts.reshape(-1, dim), normalize=True)
                per_image_text_features = model.encode_text(per_image_texts.reshape(-1, dim), normalize=True)
                per_image_group_text_features = model.encode_text(per_image_group_texts.reshape(-1, dim), normalize=True)
                label_text_features = label_text_features.reshape(bs, k, -1)
                per_image_text_features = per_image_text_features.reshape(bs, k, -1)
                per_image_group_text_features = per_image_group_text_features.reshape(bs, k, -1)
                
                # w = 0.99
                w = 1.0
                text_features = label_text_features * w + per_image_text_features * (1.0-w)/2.0 + per_image_group_text_features * (1.0-w)/2.0
                

                # Get prototype for each image based on its pseudo-label
                membank_features = []
                for i in range(len(zeroshot_classnames)):
                    membank_features_per_item = []
                    for j in range(len(zeroshot_classnames[i])):
                        zeroshot_classname = zeroshot_classnames[i][j]
                        membank_features_per_item.append(model.memory_bank[zeroshot_classname])
                    membank_features.append(torch.stack(membank_features_per_item))
                membank_features = torch.stack(membank_features) # bs, k , dim
                
                

                
                
                if membank_features.sum() != 0:
                    # if the prototype is initialized already, momentum update it
                    membank_features = F.normalize(membank_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    # common_text_features = F.normalize(common_text_features, dim=-1)
                    membank_features_text_features = membank_features + args.alpha * (text_features - membank_features) 
                else:
                    # initialize the prototype of each class as the first text feature seen
                    # membank_features_text_features = text_features
                    membank_features_text_features = label_text_features

                # detach to remove gradient to the text encoder
                membank_features_text_features = membank_features_text_features.detach() # NOTE: important to not train the text_features
                membank_features_text_features_normalized = F.normalize(membank_features_text_features, dim=-1) 
                image_features_normalized = F.normalize(image_features, dim=-1)
                
                
                losses = loss(image_features=image_features_normalized,
                            #   text_features=membank_features_text_features_normalized,
                              text_features=membank_features_text_features_normalized.squeeze(1),
                              logit_scale=logit_scale,
                            #   args=args,
                              output_dict=True)
                
               

                # Save the prototypes
                membank_features = []
                temp_bank = {}
                cnt = collections.defaultdict(int)
                for i in range(len(zeroshot_classnames)):
                    for j in range(len(zeroshot_classnames[i])):
                        if j != 0:
                            continue # only update prototype using the first text
                        zeroshot_classname = zeroshot_classnames[i][j]
                        if zeroshot_classname not in temp_bank:
                            temp_bank[zeroshot_classname] = torch.zeros_like(membank_features_text_features[i][j])
                        # Update the prototype with text features
                        text_image_features = membank_features_text_features[i][j] * (1-args.gamma) + image_features_normalized[i] * args.gamma
   
                        temp_bank[zeroshot_classname] += text_image_features
                        cnt[zeroshot_classname] += 1
                        
                for classname in temp_bank:
                    model.memory_bank[classname] = temp_bank[classname] / cnt[classname]

                        
               
                total_loss = sum(losses.values())
                
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            raise NotImplemented()
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1 :]
                        )

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
       
        
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def accuracy(output, target, topk=(1,)):
    logits, pred = output.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    accs = [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

    return accs, logits, pred.t()
    

def test_time_tuning(model, inputs, optimizer, scaler, args, reward_model=None):
    selected_idx = None
    sample_k = reward_model.sample_k
    for j in range(args.tta_step):
        with torch.cuda.amp.autocast():
            # here only play confident selection once
            if selected_idx is not None:
                output = model(inputs[selected_idx])
            else:
                output = model(inputs)
                output, selected_idx = select_confident_samples(output, args.selection_p)
                reward_model.set_image_features(inputs[selected_idx])
            bs = output.shape[0]

            # top-k sample results
            value, index = torch.topk(output, sample_k, dim=-1)
            flatten_index = index.flatten()
            # reward calculation
            clip_score = reward_model.CLIPScore(class_index=flatten_index, pairwise=False)     
            rewards = reward_model.rewards_post_process(clip_score if reward_model.process_batch else clip_score.reshape(bs, -1))

            rep_output = torch.repeat_interleave(output, sample_k, dim=0)
            all_loss = F.cross_entropy(rep_output, flatten_index, reduction='none')
            loss = torch.mean(rewards * all_loss)

            # if args.min_entropy_reg:
            #     loss = loss + args.min_entropy_w * avg_entropy(output)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def test_time_tuning_tpt(model, inputs, optimizer, scaler, args):
    args.cocoop = False
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_step):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return


def evaluate_tta(model, data, epoch, args, tokenizer, scaler, reward_model=None):
    model.eval()
    print("tta step", args.tta_step, "selection_p", args.selection_p)
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    class_names = data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"].class_names
    templates = data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"].templates
    dataloader = data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"].dataloader
    label_texts = [templates[0](class_name) for class_name in class_names]
    
    if reward_model is not None:
        reward_model.set_class_features(label_texts)
    
    model.set_class_features(label_texts)
    autocast = get_autocast(args.precision)
    
    # trainable_param = model.prompt_learner.parameters()
    # optimizer = torch.optim.AdamW(trainable_param, args.lr)
    
    trainable_param = model.parameters()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=5e-4)
    optim_state = deepcopy(optimizer.state_dict())
    
    
    
    

    
    # with torch.no_grad():
        # if model.memory_bank is not None:
        #     class_text_features = [] 
        #     for class_name in class_names:
        #         class_text_features.append(model.memory_bank[class_name])
        #     class_text_features = torch.stack(class_text_features)
        #     class_text_features = F.normalize(class_text_features, dim=1)
            
        # else:
        #     class_text_features = model.encode_text([templates[0](class_name) for class_name in class_names], normalize=True)
        
    # tta_step = args.tta_step
    top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
    total_time = 0.0
    for i, batch in tqdm(enumerate(dataloader)):

        # images, texts, metadatas = batch
        image_ids, images, target = batch
        for k in range(len(images)):
            images[k] = images[k].to(device=device, dtype=input_dtype, non_blocking=True)
        
        image = images[0]
        images = torch.cat(images, dim=0)
        # with torch.no_grad():
        #     reward_image_features = model.encode_image(images, normalize=True)
        target = target.to(args.device)
        
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)
        # model.train()
        torch.cuda.synchronize()
        start = time.time()
        test_time_tuning(model, images, optimizer, scaler, args, reward_model=reward_model)
        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        # test_time_tuning_tpt(model, images, optimizer, scaler, args)
        # model.eval()
     
        with torch.no_grad():
            with autocast():
                torch.cuda.synchronize()
                time_start = time.time()
                logits = model(image)
                torch.cuda.synchronize()
                time_end = time.time()
                total_time += time_end - time_start

        (acc1, acc5, acc10), top_logits, top_class_ids = accuracy(logits, target, topk=(1, 5, 10))
        
       
        top1 += acc1
        top5 += acc5
        top10 += acc10
        n += 1
        if n % 10 == 0:  
            print("Acc top 1, 5, 10: {:.4f}, {:.4f}, {:.4f}".format(top1 / n, top5 / n, top10 / n))
           
    print("Final acc top 1, 5, 10: {:.4f}, {:.4f}, {:.4f}".format(top1 / n, top5 / n, top10 / n))
    print("average time per image", total_time / n)
 



def extract_features(model, data, epoch, args, tokenizer):
    device = torch.device(args.device)
    model.eval()
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    class_names = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].class_names
    templates = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].templates
    dataloader = data[f"{args.zeroshot_eval_data}-{args.extract_features_split}-zero-shot-classification"].dataloader
   
    
    num_samples = 0
    
    features = {}
    os.makedirs(args.extract_features_path, exist_ok=True)
    
    logging.info("Building zero-shot classifier")
    autocast = get_autocast(args.precision)
 
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=class_names,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
    total_time = 0.0
    with torch.no_grad():
        top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
        for i, batch in tqdm(enumerate(dataloader)):

            # images, texts, metadatas = batch
            image_ids, images, target = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            target = target.to(args.device)
            
            with autocast():
                torch.cuda.synchronize()
                time_start = time.time()
                image_features = model.encode_image(images, normalize=True)
                logits = 100.0 * image_features @ classifier
                torch.cuda.synchronize()
                time_end = time.time()
                total_time += time_end - time_start
                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                (acc1, acc5, acc10), top_logits, top_class_ids = accuracy(logits, target, topk=(1, 5, 10))
                top1 += acc1
                top5 += acc5
                top10 += acc10
                n += images.size(0)
                
                
                for image_id, image_feature, top_class_id, gt_class_id, top_logit in zip(image_ids, image_features, top_class_ids, target, top_logits):
                    
                    features[image_id] = {
                        "image": image_feature.cpu().numpy(),
                        "top_class_ids": top_class_id.cpu().numpy(), 
                        "class_names": [class_names[i] for i in top_class_id.cpu().numpy()],
                        "top_logit": top_logit.cpu().numpy(),
                        "gt_classname": class_names[gt_class_id],
                        "gt_class_id": gt_class_id.item()
                    }
                    num_samples += 1
    print(f"Extracted features for {num_samples}/{args.val_num_samples} samples")
    print("Acc top 1, 5, 10: {:.4f}, {:.4f}, {:.4f}".format(top1 / n, top5 / n, top10 / n))
    save_path = os.path.join(args.extract_features_path, f"clip_features_{args.extract_features_split}.pkl")
    print("average time per image", total_time / n)
    with open(save_path, "wb") as f:
        pickle.dump(features, f)
    print("Saved features to {}".format(save_path))
                                

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
   
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts, metadata = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels)
                        + F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                        )

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {
                    **val_metrics,
                    "clip_val_loss": loss.item(),
                    "epoch": epoch,
                    "num_samples": num_samples,
                }
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, "Please install wandb."
        if "train" in data:
            dataloader = data["train"].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data["epoch"] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
