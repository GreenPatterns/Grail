import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from mycode.utils import calculate_auc, setup_features, getBestGPU
from mycode.dataset import get_snapshot_index
from mycode.attacks_gpu import initiate_attack
from torch.optim import Adam
from mycode.dgten import DGTEN


class GCNTrainer(object):
    def __init__(self, args, edges, use_GPU=True):
        self.args = args
        self.graph = edges
        if use_GPU:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                selected = None
                for i in range(torch.cuda.device_count()):
                    try:
                        cc = torch.cuda.get_device_capability(i)
                        # Require compute capability >= 7.5 (sm_75+)
                        if cc[0] > 7 or (cc[0] == 7 and cc[1] >= 5):
                            selected = i
                            break
                    except Exception:
                        continue
                if selected is not None:
                    self.device = torch.device(f'cuda:{selected}')
                else:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device("cpu")
        self.best_loss = float('inf')
        self.best_model_path = getattr(args, 'best_model_path', None)
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
        self.performance_history = []
        self.global_start_time = time.time()
        self.setup_logs()
        self.attack = None

    def setup_logs(self):
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    def setup_dataset(self):
        self.index_list = get_snapshot_index(self.args.time_slots, data_path=self.args.data_path, homogeneous_edges=self.args.homogeneous_edges)
        index_t = self.index_list[self.args.train_time_slots-1] + 1
        self.train_edges = self.graph['edges'][:index_t]
        self.train_labels = self.graph['labels'][:index_t]
        train_set = set(list(self.train_edges.flatten()))

        if self.args.single_prediction:
            enable_task3 = hasattr(self.args, 'enable_task3') and self.args.enable_task3
            index_t1 = self.index_list[self.args.train_time_slots] + 1
            self.test_edges = self.graph['edges'][index_t:index_t1]
            self.test_labels = self.graph['labels'][index_t:index_t1]

            if enable_task3:
                self.unobs, self.y_test_unobs = [], []
                index_t_1 = self.index_list[self.args.train_time_slots-2]
                train_pre = set(list(self.train_edges[:index_t_1].flatten()))
                train_t = set(list(self.train_edges[index_t_1:].flatten()))
                for i in range(len(self.test_edges)):
                    tr, te = self.test_edges[i][0], self.test_edges[i][1]
                    if (tr in train_t and tr not in train_pre and te not in train_t and te in train_pre) or \
                       (te in train_t and te not in train_pre and tr not in train_t and tr in train_pre) or \
                       (tr in train_t and te in train_t and tr not in train_pre and te not in train_pre):
                        self.unobs.append(self.test_edges[i])
                        self.y_test_unobs.append(self.test_labels[i])
                self.unobs = np.array(self.unobs) if self.unobs else np.array([])
                self.y_test_unobs = np.array(self.y_test_unobs) if self.y_test_unobs else np.array([])
                self.obs_test_edges, self.obs_test_labels = np.array([]), np.array([])
            else:
                self.obs_test_edges, self.obs_test_labels = [], []
                for i in range(len(self.test_edges)):
                    if self.test_edges[i][0] in train_set and self.test_edges[i][1] in train_set:
                        self.obs_test_edges.append(self.test_edges[i])
                        self.obs_test_labels.append(self.test_labels[i])
                self.obs_test_edges = np.array(self.obs_test_edges)
                self.obs_test_labels = np.array(self.obs_test_labels)
                self.unobs, self.y_test_unobs = np.array([]), np.array([])
        else:
            index_pre = self.index_list[self.args.train_time_slots - 1] + 1
            index_lat = self.index_list[self.args.train_time_slots + 2] + 1
            self.test_edges = self.graph['edges'][index_pre:index_lat]
            self.test_labels = self.graph['labels'][index_pre:index_lat]
            self.obs_test_edges, self.obs_test_labels = [], []
            for i in range(len(self.test_edges)):
                if self.test_edges[i][0] in train_set and self.test_edges[i][1] in train_set:
                    self.obs_test_edges.append(self.test_edges[i])
                    self.obs_test_labels.append(self.test_labels[i])
            self.obs_test_edges = np.array(self.obs_test_edges)
            self.obs_test_labels = np.array(self.obs_test_labels)

        dim = 16
        self.X = setup_features(self.graph, self.args.seed, dim=dim)
        self.num_labels = int(np.shape(self.train_labels)[1])
        self.y = torch.from_numpy(self.train_labels[:, 1]).type(torch.long).to(self.device)
        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.train_labels = torch.from_numpy(np.array(self.train_labels, dtype=np.float32)).type(torch.float).to(self.device)
        self.X = torch.from_numpy(self.X).to(self.device)

        if self.args.attack:
            self._precompute_poisoned_snapshots()
            self.y_train = self.train_labels_final[:, 1].long().to(self.device)
            self.y_test = self.test_labels_final[:, 1].long().to(self.device)
        else:
            self.train_edges_final = self.train_edges
            self.train_labels_final = self.train_labels
            self.test_edges_final = self.train_edges
            self.test_labels_final = self.train_labels
            self.y_train = self.y
            self.y_test = self.y

    def _precompute_poisoned_snapshots(self):
        num_snaps = self.args.train_time_slots
        index0 = 0

        poisoned_edges_list = []
        poisoned_labels_list = []
        clean_edges_list = []
        clean_labels_list = []

        for i in range(num_snaps):
            end = self.index_list[i] + 1
            lim = slice(index0, end)

            snap_edges = self.train_edges[:, lim]
            snap_labels = self.train_labels[lim, :]

            clean_edges_list.append(snap_edges)
            clean_labels_list.append(snap_labels)

            attack_type_to_use = None
            current_victim_pct = getattr(self.args, 'victim_percentage', 0.10)

            if self.args.attack_type == 'bad_mouthing':
                attack_type_to_use = 'bad_mouthing'
            elif self.args.attack_type == 'good_mouthing':
                attack_type_to_use = 'good_mouthing'
            elif self.args.attack_type == 'on-off':
                if i % 2 == 0:
                    attack_type_to_use = 'bad_mouthing'
            elif self.args.attack_type == 'slow_poisoning':
                attack_type_to_use = 'slow_poisoning'
                current_victim_pct = self.args.victim_percentage
            elif self.args.attack_type in {
                'camouflage',
                'adaptive_stealth',
                'adaptive_gradient',
                'low_budget_stealth',
                'ballot_stuffing',
                'sybil_boosting',
            }:
                attack_type_to_use = self.args.attack_type

            if attack_type_to_use:
                target_edges = None
                target_labels = None
                if getattr(self.args, "attack_target_future", True) and len(self.obs_test_edges) > 0:
                    target_edges = torch.from_numpy(
                        np.array(self.obs_test_edges, dtype=np.int64).T
                    ).type(torch.long).to(self.device)
                    target_labels = torch.from_numpy(
                        np.array(self.obs_test_labels, dtype=np.float32)
                    ).type(torch.float).to(self.device)
                poisoned_snap_edges, poisoned_snap_labels = initiate_attack(
                    original_edges=snap_edges,
                    original_labels=snap_labels,
                    attack_type=attack_type_to_use,
                    attack_percentage=float(getattr(self.args, "attack_percentage", 1.0)),
                    victim_percentage=current_victim_pct,
                    verbose=False,
                    seed=self.args.seed,
                    snapshot_idx=i,
                    total_snapshots=self.args.train_time_slots,
                    target_edges=target_edges,
                    target_labels=target_labels)
            else:
                poisoned_snap_edges, poisoned_snap_labels = snap_edges, snap_labels

            poisoned_edges_list.append(poisoned_snap_edges)
            poisoned_labels_list.append(poisoned_snap_labels)
            index0 = end

        self.train_edges_final = torch.cat(poisoned_edges_list, dim=1)
        self.train_labels_final = torch.cat(poisoned_labels_list, dim=0)
        self.test_edges_final = torch.cat(clean_edges_list, dim=1)
        self.test_labels_final = torch.cat(clean_labels_list, dim=0)

        poisoned_index_list = []
        cumsum = 0
        for edges in poisoned_edges_list:
            cumsum += edges.shape[1]
            poisoned_index_list.append(cumsum - 1)
        self.poisoned_index_list = poisoned_index_list

    def check_early_stopping(self, current_performance):
        if not current_performance or current_performance[1] is None:
            return False
        current_auc = current_performance[1]
        self.performance_history.append(current_auc)
        if len(self.performance_history) < self.early_stopping_patience + 1:
            return False
        recent_performances = self.performance_history[-(self.early_stopping_patience + 1):]
        best_recent_auc = max(recent_performances[:-1])
        if current_auc > best_recent_auc:
            self.early_stopping_counter = 0
            return False
        self.early_stopping_counter += 1
        return self.early_stopping_counter >= self.early_stopping_patience

    def create_and_train_model(self, startmsg=None):
        startmsg = " Training " + startmsg if startmsg is not None else "Training "
        index_list_to_use = self.index_list
        if self.args.attack and hasattr(self, 'poisoned_index_list'):
            index_list_to_use = self.poisoned_index_list
        self.model = DGTEN(self.device, self.args, self.X, self.num_labels, index_list_to_use).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        pbar = tqdm(range(self.args.epochs), desc=startmsg)
        for epoch in pbar:
            self.model.train()
            start_time = time.time()
            optimizer.zero_grad()

            train_loss, temporal_out, mu = self.model(
                self.train_edges_final,
                self.y_train,
                self.train_labels_final,
                adversarial=True,
            )
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            pbar.set_postfix({"Saved at ": f"{self.best_loss:.4f}", "TrLoss": f"{train_loss:.4f}"})
            self.logs["training_time"].append([epoch + 1, time.time() - start_time])

            current_performance = self.test_model(epoch, self.model)
            # if self.check_early_stopping(current_performance) and False:
            #     print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
            #     pbar.close()
            #     break

        self.logs["training_time"].append(["Total", time.time() - self.global_start_time])

    def test_model(self, epoch, model_testing):
        model_testing.eval()
        return_values = [None, None, None, None, None, None]
        saved_index_list = None
        if self.args.attack and hasattr(self, 'poisoned_index_list'):
            saved_index_list = model_testing.index_list
            model_testing.index_list = self.index_list
        with torch.no_grad():
            _, self.node_embedding_z, _ = model_testing(self.test_edges_final, self.y_test, self.test_labels_final)
        enable_task3 = hasattr(self.args, 'enable_task3') and self.args.enable_task3

        if enable_task3 and len(self.unobs) > 0:
            unobs_edges = torch.from_numpy(np.array(self.unobs, dtype=np.int64).T).type(torch.long).to(self.device)
            unobs_z = torch.cat((self.node_embedding_z[unobs_edges[0, :], :], self.node_embedding_z[unobs_edges[1, :], :]), 1)
            unobs_scores = torch.mm(unobs_z, model_testing.regression_weights.to(self.device))
            unobs_predictions = F.softmax(unobs_scores, dim=1)
            res = calculate_auc(unobs_predictions, self.y_test_unobs)
            self.logs["performance"].append([epoch + 1] + list(res))
            return_values = list(res)
        elif len(self.obs_test_edges) > 0:
            score_edges = torch.from_numpy(np.array(self.obs_test_edges, dtype=np.int64).T).type(torch.long).to(self.device)
            test_z = torch.cat((self.node_embedding_z[score_edges[0, :], :], self.node_embedding_z[score_edges[1, :], :]), 1)
            scores = torch.mm(test_z, model_testing.regression_weights.to(self.device))
            predictions = F.softmax(scores, dim=1)
            res = calculate_auc(predictions, self.obs_test_labels)
            self.logs["performance"].append([epoch + 1] + list(res))
            return_values = list(res)
        else:
            self.logs["performance"].append([epoch + 1, None, None, None, None, None, None])
        if saved_index_list is not None:
            model_testing.index_list = saved_index_list
        return return_values
