import os
import sys
import subprocess
import argparse
from time import gmtime, strftime
import json

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--arch', type=str, help='model architecture',
                        choices=['ltp-base', 'ltp-large'])
    parser.add_argument('--task', type=str, help='finetuning task',
                        choices=['RTE', 'SST2', 'MNLI', 'QNLI', 'QQP', 'MRPC', 'STSB'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--restore', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--lambda_threshold', type=float, default=0)
    parser.add_argument('--weight_decay_threshold', type=float, default=None)
    parser.add_argument('--lr_threshold', type=float, default=None)
    parser.add_argument('--masking_mode', type=str, 
                        choices=['hard', 'soft', 'mixed'], default='hard') 
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--save_all', action='store_true') 
    parser.add_argument('--no_load', action='store_true') 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--task_name', type=str, default='RTE')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--logging_strategy', type=str, default='epoch')
    parser.add_argument('--save_strategy', type=str, default='epoch')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.98)
    parser.add_argument('--adam_epsilon', type=float, default=1e-06)
    parser.add_argument('--warmup_steps', type=int, default=122)
    parser.add_argument('--lr_scheduler_type', type=str, default='polynomial')
    parser.add_argument('--load_best_model_at_end', type=bool, default=False)
    parser.add_argument('--metric_for_best_model', type=str, default='eval_accuracy')
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--half_precision_backend', type=str, default='auto')
    parser.add_argument('--dataloader_num_workers', type=int, default=1)
    parser.add_argument('--model_name_or_path', type=str, default='')
    parser.add_argument('--num_train_epochs', type=int, default=10)


    args = parser.parse_args()
    return args

args = arg_parse()

with open(os.path.join(args.restore, "config.json")) as f:
    config = json.load(f)

prune_mode = config['prune_mode']
scoring_mode = None
if prune_mode == 'topk':
    rate = config['token_keep_rate']
elif prune_mode in ['absolute_threshold', 'rising_threshold']:
    rate = config['final_token_threshold']
    # if prune_mode == 'absolute_threshold':
        # scoring_mode = config['scoring_mode']
else:
    rate = None

if args.task is None:
    print('please specify --task')
    sys.exit()

if args.arch is None:
    print('please specify --arch')
    sys.exit()

DEFAULT_OUTPUT_DIR = 'checkpoints'

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
max_epochs = str(args.epoch)

task_specs = {
    'RTE' : {
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'MRPC' : {
        'lr': '2e-5',
        'metric': 'eval_combined_score',
    },
    'COLA' : {
        'metric': 'eval_matthews_correlation',
        'lr': '4e-5',
    },
    'STSB' : {
        'lr': '4e-5',
        'metric': 'eval_combined_score',
    },
    'SST2' : {
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'QNLI' : {
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'QQP' : {
        'lr': '4e-5',
        'metric': 'eval_combined_score',
    },
    'MNLI' : {
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
}

model_path = args.arch
if args.restore is not None:
    model_path = args.restore

if args.eval and args.restore is None:
    print('Please specify --restore for the eval mode')
    sys.exit()
    
spec = task_specs[args.task]
lr = spec['lr']
is_large = ('large' in args.arch)
assert 'metric' in spec, 'please specify metric for %s' % args.task
metric = spec['metric']

# set learning rate
if args.lr:
    lr = str(args.lr)
    print('lr is set as %s' % lr)

#output_dir = args.output_dir
#if output_dir is None:
output_dir = DEFAULT_OUTPUT_DIR  + ('/large' if is_large else '/base')
task = args.task

#output_file = '%s/%s/tkr_%s/%s' % (args.task, prune_mode, rate, lr)
if args.output_dir is None:
    if prune_mode == 'topk':
        #output_file = '%s/%s/rate_%s/lr_%s' % (args.task, prune_mode, rate, lr)
        output_file = os.path.join(args.restore, f"topk/lr_{lr}")
        output_path = output_file
    else:
        assert prune_mode == 'absolute_threshold'
        if args.masking_mode == 'soft':
            _temperature = args.temperature if args.temperature is not None else 1e-3
            output_file = f"{args.task}/{prune_mode}/rate_{rate}/temperautre_{_temperature}/lambda_{args.lambda_threshold}/lr_{lr}"
            output_path = os.path.join(output_dir, output_file)
        elif args.masking_mode == 'hard':
            output_file = os.path.join(args.restore, f"hard/lr_{lr}")
            output_path = output_file
        else:
            raise NotImplementedError

else:
    output_file = '%s/%s/rate_%s/lambda_%s/%s/tlr_%s/lr_%s/%s' % \
            (args.task, prune_mode, rate, args.lambda_threshold, args.output_dir, 
             args.lr_threshold, lr, args.masking_mode)
    output_path = os.path.join(output_dir, output_file)

print('output path: ', output_path)


if 'ltp' in args.arch:
    run_file = 'examples/pytorch/text-classification/run_glue_ltp.py'
else:
    run_file = 'examples/text-classification/run_glue.py'

subprocess_args = [
    'python', run_file,
    '--model_name_or_path', model_path,
    '--task_name', args.task,
    '--do_eval',
    '--max_seq_length', '128',
    '--per_device_train_batch_size', str(args.bs),
    '--per_device_eval_batch_size', str(args.bs),
    '--masking_mode', args.masking_mode,
    '--seed', str(args.seed),
    '--eval_steps', str(args.eval_steps),
    '--output_dir', str(args.output_dir),
    '--save_steps', str(args.save_steps),
    '--lambda_threshold', str(args.lambda_threshold),
    '--temperature', str(args.temperature),
    '--fp16', str(args.fp16),
    '--evaluation_strategy', str(args.evaluation_strategy),
    '--logging_strategy', str(args.logging_strategy),
    '--save_strategy', str(args.save_strategy),
    '--max_seq_length', str(args.max_seq_length),
    '--per_device_train_batch_size', str(args.per_device_train_batch_size),
    '--per_device_eval_batch_size', str(args.per_device_eval_batch_size),
    '--weight_decay', str(args.weight_decay),
    '--adam_beta1', str(args.adam_beta1),
    '--adam_beta2', str(args.adam_beta2),
    '--adam_epsilon', str(args.adam_epsilon),
    '--warmup_steps', str(args.warmup_steps),
    '--lr_scheduler_type', str(args.lr_scheduler_type),
    '--load_best_model_at_end', str(args.load_best_model_at_end),
    '--metric_for_best_model', str(args.metric_for_best_model),
    '--optim', str(args.optim),
    '--half_precision_backend', str(args.half_precision_backend),
    '--dataloader_num_workers', str(args.dataloader_num_workers),
    '--num_train_epochs', str(args.num_train_epochs),
    ]

# Training mode
if not args.eval:
    subprocess_args.append('--do_train')
    if args.save_steps is None:
        subprocess_args += ['--evaluation_strategy', 'epoch'] 
        subprocess_args += ['--logging_strategy', 'epoch'] 
    else:
        # subprocess_args += ['--evaluation_strategy', 'steps'] 
        subprocess_args += ['--eval_steps', str(args.save_steps)]
        # subprocess_args += ['--logging_strategy', 'steps'] 
        subprocess_args += ['--logging_steps', str(args.save_steps)]
    subprocess_args += [
                   '--metric_for_best_model', metric,
                   '--learning_rate', lr, 
                   '--num_train_epochs', max_epochs, 
                   '--output_dir', output_path, 
                   ]
    if not args.no_load:
        subprocess_args += ['--load_best_model_at_end', 'True']
    if not args.save_all:
        subprocess_args += ['--save_total_limit', '3']

    if args.lr_threshold is not None:
        subprocess_args += ['--lr_threshold', str(args.lr_threshold)]

    if args.weight_decay_threshold is not None:
        subprocess_args += ['--weight_decay_threshold', str(args.weight_decay_threshold)]

    if args.lambda_threshold is not None:
        subprocess_args += ['--lambda_threshold', str(args.lambda_threshold)]

    if args.temperature is not None:
        subprocess_args += ['--temperature', str(args.temperature)]

# Eval-only mode
else:
    subprocess_args += ['--output_dir', '/tmp/temp', '--overwrite_output_dir']

print(subprocess_args)
subprocess.call(subprocess_args)