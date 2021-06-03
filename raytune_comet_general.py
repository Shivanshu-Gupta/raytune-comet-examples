import os
import time
import pandas as pd
import comet_ml
from comet_ml import Experiment

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# os.environ['COMET_PROJECT_NAME'] = 'humor-1'
# os.environ['COMET_MODE'] = 'ONLINE'

def evaluation_fn(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100)**(-1) + height * 0.1

# ref: https://docs.ray.io/en/master/tune/api_docs/trainable.html
def easy_objective(config, checkpoint_dir: str = None, comet=False):
    # setup comet experiment and log parameters
    if comet:
        # ref: https://www.comet.ml/docs/python-sdk/Experiment
        experiment = Experiment(project_name='raytune_comet_example')
        experiment.add_tag(config['search_name'])
        experiment.log_parameters(config)

    # Hyperparameters
    width, height = config["width"], config["height"]

    for epoch_num in range(config["num_epochs"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(epoch_num, width, height)

        # Feed the score back back to Tune.
        tune.report(iterations=epoch_num, mean_loss=intermediate_score)

        # log metrics to comet
        if comet:
            # ref: https://www.comet.ml/docs/python-sdk/Experiment
            experiment.log_metrics({"mean_loss": intermediate_score},
                                   epoch=epoch_num)

        # Save checkpoint
        with tune.checkpoint_dir(step=epoch_num) as checkpoint_dir:
            pass

def get_hp_space(idx, num_epochs, hyperopt=False):
    # ref: https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api
    choices = tune.choice if hyperopt else tune.grid_search
    hp_spaces = [
        lambda: dict(width=tune.uniform(0, 20),
                     height=tune.uniform(-100, 100),
                     lr=tune.loguniform(1e-3, 1e-1),
                     activation=choices(["relu", "tanh"])),
        lambda: dict(width=choices([5, 10, 20]),
                     height=choices([-100, -50, 0, 50, 100]),
                     lr=choices([1e-3, 1e-2, 1e-1]),
                     activation=choices(["relu", "tanh"]))
        ]
    hp_space = hp_spaces[idx]()
    hp_space['num_epochs'] = num_epochs     # constant parameter - not tuned
    return hp_space

def load_analysis(ray_dir, search_name, metric='mean_loss', mode='min'):
    """ Use this to load analyse the results of a tune run later. """
    experiment_dir = os.path.join(ray_dir, search_name)
    if not os.path.exists(experiment_dir):
        return None, None, None, None

    from ray.tune import Analysis
    analysis = Analysis(experiment_dir=experiment_dir,
                        default_metric=metric, default_mode=mode)

    df: pd.DataFrame = analysis.dataframe()
    best_config = analysis.get_best_config()

    if mode == 'min': best_results = df.loc[df[metric].idxmin()].to_dict()
    else: best_results = df.loc[df[metric].idxmax()].to_dict()

    return analysis, df, best_config, best_results


if __name__ == "__main__":
    # python raytune_comet.py --search_name smoke --max_num_epochs=20 --smoke_test --comet
    # python raytune_comet.py --search_name random --max_num_epochs=20 --num_samples=4 --comet
    # python raytune_comet.py --search_name hyperopt --max_num_epochs=20 --num_samples=4 --hyperopt --comet
    # python raytune_comet.py --search_name hpspace1 --hp_space_idx=1 --max_num_epochs=20 --num_samples=1 --comet

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_name', type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Finish quickly for testing")
    parser.add_argument("--server_address", type=str, default='0.0.0.0',
                        help="The address of server to connect to if using Ray Client.")
    parser.add_argument('--hp_space_idx', type=int, default=0,
                        help='hyperparameter space to search')
    parser.add_argument('--num_samples', type=int, default=10,
                        help="number of samples per grid point. "
                             "(num_trials = num_samples * num_grid_points)")
    parser.add_argument('--max_num_epochs', type=int, default=10,
                        help='maximum number of training epochs')
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--hyperopt', action='store_true',
                        help='Use hyperopt for search suggestions')
    parser.add_argument('--comet', action='store_true',
                        help='Use comet for logging/tracking.')
    parser.add_argument('--random_seed', type=int, default=42)
    args, _ = parser.parse_known_args()

    ray.init(dashboard_host=args.server_address)        # need to do `pip install ray[default]` to use dashboard

    config = get_hp_space(idx=args.hp_space_idx, num_epochs=args.max_num_epochs,
                          hyperopt=args.hyperopt)

    # ref: https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    scheduler = ASHAScheduler(
        max_t=args.max_num_epochs,
        grace_period=5,
        reduction_factor=2)

    search_alg = None
    if args.hyperopt:
        # ref: https://docs.ray.io/en/master/tune/api_docs/suggestion.html
        guesses = [
            dict(width=10, height=50, lr=1e-2, activation="relu")
            ]
        search_alg = HyperOptSearch(metric="mean_loss", mode="min",
                                    random_state_seed=args.random_seed,
                                    points_to_evaluate=guesses)
    if args.search_name:
        config['search_name'] = args.search_name
    else:
        config['search_name'] = f"search-{args.hp_space_idx}"
    print(config)

    # ref: https://docs.ray.io/en/master/tune/api_docs/execution.html#tune-run
    analysis = tune.run(
        # easy_objective,
        tune.with_parameters(easy_objective, comet=args.comet),
        name=config['search_name'],
        config=config,
        resources_per_trial={"cpu": 2, 'gpu': args.gpus_per_trial},
        metric="mean_loss",
        mode="min",
        num_samples=args.num_samples if not args.smoke_test else 1,
        stop={"training_iteration": args.max_num_epochs},
        scheduler=scheduler,
        search_alg=search_alg,
        local_dir="ray_results/basic",
        log_to_file=True,
        verbose=2
        )
    df = analysis.dataframe()
    best_results = df.loc[df['mean_loss'].idxmin()].to_dict()
    print("Best hyperparameters found were: ", analysis.best_config)
