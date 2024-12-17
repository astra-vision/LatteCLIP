import logging

import torch
from tqdm import tqdm
import torch.nn.functional as F

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
)
from .precision import get_autocast

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
        for _, images, target in tqdm(dataloader, unit_scale=args.batch_size):

            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = (
                    output["image_features"] if isinstance(output, dict) else output[0]
                )
                logits = 100.0 * image_features @ classifier

            # measure accuracy
            acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
            top1 += acc1
            top5 += acc5
            top10 += acc10
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    top10 = top10 / n
    return top1, top5, top10


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if args.method == "reclip":
        results = {}
        t_model = model.t_model
        v_model = model.v_model
        for task in data:
            if "val-zero-shot-classification" in task:
                autocast = get_autocast(args.precision)
                input_dtype = get_input_dtype(args.precision)
                dataloader = data[task].dataloader
                class_names = data[task].class_names
                
                with torch.no_grad():
                    top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
                    for _, images, target in tqdm(dataloader, unit_scale=args.batch_size):

                        images = images.to(device=args.device, dtype=input_dtype)
                        target = target.to(args.device)

                        with autocast():
                            t_logits, _ = t_model(images, class_names)
                            v_logits, _ = v_model(images)
                            logits = 0.5 * (t_logits + v_logits)
                     
                        # measure accuracy
                        acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
                        top1 += acc1
                        top5 += acc5
                        top10 += acc10
                        n += images.size(0)

                top1 = top1 / n
                top5 = top5 / n
                top10 = top10 / n
            
            
                results[f"{task}-val-top1"] = top1
                results[f"{task}-val-top5"] = top5
                results[f"{task}-val-top10"] = top10
    else:
        if args.zeroshot_frequency == 0:
            return {}
        if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
            return {}
        if args.distributed and not args.horovod:
            model = model.module

        logging.info("Starting zero-shot classification evaluation.")
        if tokenizer is None:
            tokenizer = get_tokenizer(args.model)

        if args.zeroshot_eval_data is not None:
            class_names = data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"].class_names
            templates = data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"].templates
        else:
            raise ValueError("No zero-shot evaluation data provided.")
    

        logging.info("Building zero-shot classifier")
        autocast = get_autocast(args.precision)
        
        with autocast():
            if args.method == "flyp":
            # if True:
                classifier = build_zero_shot_classifier(
                    model,
                    tokenizer=tokenizer,
                    classnames=class_names,
                    templates=templates,
                    num_classes_per_batch=10,
                    device=args.device,
                    use_tqdm=True,
                )
            elif args.method == "rlcf":
                classifier = build_zero_shot_classifier(
                    model,
                    tokenizer=tokenizer,
                    classnames=class_names,
                    templates=templates,
                    num_classes_per_batch=10,
                    device=args.device,
                    use_tqdm=True,
                )
            else:
                # NOTE: create classifier from prototypes  
                mem_classifier = []
                for i, classname in enumerate(class_names):
                    mem_classifier.append(model.memory_bank[classname])
                mem_classifier = torch.stack(mem_classifier)
                mem_classifier = F.normalize(mem_classifier, dim=1)
                classifier = mem_classifier.T
        

        logging.info("Using classifier")
        results = {}
        if "imagenet-val" in data:
            top1, top5 = run(model, classifier, data["imagenet-val"].dataloader, args)
            results["imagenet-zeroshot-val-top1"] = top1
            results["imagenet-zeroshot-val-top5"] = top5
        if "imagenet-v2" in data:
            top1, top5 = run(model, classifier, data["imagenet-v2"].dataloader, args)
            results["imagenetv2-zeroshot-val-top1"] = top1
            results["imagenetv2-zeroshot-val-top5"] = top5
        for task in data:
            if "val-zero-shot-classification" in task:
                top1, top5, top10 = run(model, classifier, data[task].dataloader, args)
                results[f"{task}-val-top1"] = top1
                results[f"{task}-val-top5"] = top5
                results[f"{task}-val-top10"] = top10
                

    logging.info("Finished zero-shot imagenet.")

    return results
