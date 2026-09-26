import gc
import torch
import os
import numpy as np
import datetime as dt
from mycode.gcn3 import GCNTrainer
from mycode.args import parameter_parser
from mycode.utils import print_config, read_graph, best_printer

NUM_EXPERIMENTS = 1
METRICS_COLUMNS = [
    "Timestep", "Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"
]
BEST_EPOCH_METRICS_IDX = [1, 2, 3, 6]
BEST_DATA_METRICS_IDX = [2, 3, 4, 7]


def train_and_evaluate_snapshot(args, graph, time_step_idx):
    trainer = GCNTrainer(args, graph, use_GPU=True)
    trainer.setup_dataset()
    start_message = f"TrainSnapshots = {args.train_time_slots}"
    trainer.create_and_train_model(startmsg=start_message)

    best_epoch_data = None
    best_epoch_score = -1.0
    for epoch_log in trainer.logs["performance"][1:]:
        if any(epoch_log[i] is None for i in BEST_EPOCH_METRICS_IDX):
            continue
        current_score = sum(float(epoch_log[i]) for i in BEST_EPOCH_METRICS_IDX)
        if current_score > best_epoch_score:
            best_epoch_score = current_score
            best_epoch_data = list(epoch_log)

    if best_epoch_data:
        training_time = trainer.logs["training_time"][-1][1]
        best_epoch_data.insert(0, time_step_idx)
        best_epoch_data.append(training_time)
        print(f"\t" + ", ".join(f"{c}:{v:.4f}" for c, v in zip(METRICS_COLUMNS[2:], best_epoch_data[2:])))
    else:
        print(f"\tNo valid epoch found for time step {time_step_idx}.")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return best_epoch_data


def run_single_experiment(args, graph, experiment_idx, num_time_steps):
    print(f"\n--- Experiment {experiment_idx + 1}/{NUM_EXPERIMENTS} ---")
    best_results_for_experiment = [METRICS_COLUMNS]
    args.train_time_slots = 7 if args.attack else 2

    for t in range(num_time_steps):
        time_step_number = t + 2
        best_epoch_data = train_and_evaluate_snapshot(args, graph, time_step_number)
        if best_epoch_data and args.enable_task3:
            if all(best_epoch_data[i] != 0.0 for i in BEST_DATA_METRICS_IDX):
                best_results_for_experiment.append(best_epoch_data)
        else:
            best_results_for_experiment.append(best_epoch_data)
        args.train_time_slots += 1

    valid_results = [r for r in best_results_for_experiment[1:] if r is not None]
    if len(valid_results) > 0:
        results_matrix = np.array(valid_results, dtype=np.float64)
        analyze = results_matrix[:, 1:]
        average = np.mean(analyze, axis=0)
        print(f"\nExp {experiment_idx + 1} Mean: " + ", ".join(f"{c}:{v:.4f}" for c, v in zip(METRICS_COLUMNS[2:], average[1:])))
        return average
    else:
        print(f"\nExp {experiment_idx + 1} had no valid results.")
        return None


def aggregate_and_print_results(all_experiment_results):
    print("--- Avg of all experiments statistics---")
    valid_results = [res for res in all_experiment_results if res is not None]
    if not valid_results:
        print("No valid results to aggregate.")
        return
    results_matrix = np.array(valid_results)
    mean = np.mean(results_matrix, axis=0)
    maxi = np.amax(results_matrix, axis=0)
    mini = np.amin(results_matrix, axis=0)
    std = np.std(results_matrix, axis=0)
    print("mean, maxi, mini, std")
    header = ["Epoch", 'MCC', "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"]
    best_printer([header, mean, maxi, mini, std])


def main():
    print("\n================> GDTE <=================")
    args = parameter_parser(
        dataset='otc',
        pred_snap='single',
        attack=False,
        atype='bad_mouthing',
        victim_percentage=0.10,
        homogeneous_edges=False,
        lambda_temp=0.01
    )
    print_config(args)
    # timestamp = dt.datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
    # args.best_model_path = f"test_models/best_model_{timestamp}.pth"
    # os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
    graph = read_graph(args)
    num_time_steps = args.time_slots - 2 if args.single_prediction else args.time_slots - 4
    num_time_steps = 1 if args.attack else num_time_steps
    all_experiment_results = []
    for exp_idx in range(NUM_EXPERIMENTS):
        experiment_avg = run_single_experiment(args, graph, exp_idx, num_time_steps)
        if experiment_avg is not None:
            all_experiment_results.append(experiment_avg)
    print_config(args)
    aggregate_and_print_results(all_experiment_results)


if __name__ == "__main__":
    main()
