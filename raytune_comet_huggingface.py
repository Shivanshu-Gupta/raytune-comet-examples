# ref: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb

import os
import comet_ml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'raytune_comet_huggingface'
os.environ['COMET_MODE'] = 'ONLINE'

from argparse import ArgumentParser
import numpy as np
import pandas as pd

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# python raytune_comet_huggingface.py -o --tiny
# python raytune_comet_huggingface.py -o --tiny --gpus_per_trial=1

parser = ArgumentParser()
parser.add_argument('--tiny', action='store_true',
                    help='Use only 1000 samples of train set for faster search.')
parser.add_argument('--num_samples', type=int, default=10,
                    help="number of samples per grid point."
                         "(num_trials = num_samples * num_grid_points)")
parser.add_argument('--max_num_epochs', type=int, default=10,
                    help='maximum number of training epochs')
parser.add_argument('--gpus_per_trial', type=int, default=0)
parser.add_argument('--hyperopt', action='store_true',
                    help='Use hyperopt for search suggestions')
parser.add_argument('--comet', action='store_true',
                    help='Use comet for logging/tracking.')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('-o', '--overwrite', action='store_true')
parser.add_argument('--evaluate_best', action='store_true')
cmd_args = parser.parse_args()

task = "qqp"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)
print(dataset)
print(metric)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
encoded_dataset = dataset.map(lambda examples: tokenizer(examples["question1"],
                                                         examples["question2"],
                                                         truncation=True),
                              batched=True)

metric_name = "accuracy"
direction = "maximise"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    print(metric.compute(predictions=predictions, references=labels))
    return metric.compute(predictions=predictions, references=labels)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

train_dataset = encoded_dataset['train']
if cmd_args.tiny:
    train_dataset = encoded_dataset["train"][:1000]

args = TrainingArguments(
    output_dir="test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=cmd_args.max_num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    overwrite_output_dir = cmd_args.overwrite,
    report_to = 'comet_ml' if cmd_args.comet else "none",
    skip_memory_metrics=True        # this is important
)

trainer = Trainer(
    model_init=model_init,      # instead of model=
    args=args,
    train_dataset=train_dataset,
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

scheduler = ASHAScheduler(
    max_t=args.num_train_epochs,
    grace_period=5,
    reduction_factor=2)

search_alg = None
if cmd_args.hyperopt:
    search_alg = HyperOptSearch(metric=f'eval_{metric_name}', mode=direction[:3],
                                random_state_seed=args.seed)

tune_args = dict(
    name=f'qqp',
    local_dir='ray_results/qqp/',
    resources_per_trial={"cpu": 2, 'gpu': cmd_args.gpus_per_trial},
    scheduler=scheduler,
    search_alg=search_alg,
    log_to_file=True,
    metric=f'eval_{metric_name}',       # Huggingface autmotically prepends 'eval_' to metrics for eval set.
    mode=direction[:3]
)
best_run = trainer.hyperparameter_search(n_trials=cmd_args.num_samples,
                                         direction=direction,
                                         backend="ray",
                                         compute_objective=lambda metrics: metrics[f'eval_{metric_name}']
                                         **tune_args)

print(best_run)
if cmd_args.evaluate_best:
    trainer.args.disable_tqdm = False
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(eval_metrics)
    test_metrics = trainer.evaluate(encoded_dataset['test'], metric_key_prefix='test')
    print(test_metrics)
