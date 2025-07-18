import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from pprint import pprint
from pathlib import Path

class MulticlassMetricsCalculator:
    def __init__(
        self, ground_truth, probabilities, n_bootstraps=1000, random_seed=None
    ):
        """
        初始化多分类指标计算器

        参数:
        ground_truth (pd.Series): 真实标签
        probabilities (pd.Series): 预测概率，每个元素是各类别的概率列表
        n_bootstraps (int): Bootstrap采样次数，默认为1000
        random_seed (int): 随机种子，确保结果可复现
        """
        self.ground_truth = ground_truth.values
        self.probabilities = np.vstack(probabilities.values)
        self.n_bootstraps = n_bootstraps
        self.classes = sorted(np.unique(self.ground_truth))
        self.n_classes = len(self.classes)
        self.n_samples = len(self.ground_truth)
        self.random_seed = random_seed

        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)

        # 预先计算所有Bootstrap索引
        self.bootstrap_indices = [
            np.random.choice(self.n_samples, self.n_samples, replace=True)
            for _ in range(n_bootstraps)
        ]

        # 二值化真实标签
        self.y_true_bin = label_binarize(self.ground_truth, classes=self.classes)

        # 生成预测标签
        self.y_pred = np.argmax(self.probabilities, axis=1)

        # 创建缓存
        self._auroc_cache = {}
        self._auprc_cache = {}

    def _calculate_ci(self, values):
        """计算95%置信区间"""
        alpha = 100 * 0.05 / 2
        return (np.percentile(values, alpha), np.percentile(values, 100 - alpha))

    def calculate_auroc(self, avg=None):
        """计算AUROC及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_auroc_full"):
            if avg is None:
                return self._auroc_full["per_class"]
            return self._auroc_full[avg]

        # 原始计算
        class_aurocs = []
        for i in range(self.n_classes):
            auc_score = roc_auc_score(self.y_true_bin[:, i], self.probabilities[:, i])
            class_aurocs.append(auc_score)

        macro_auc = np.mean(class_aurocs)
        micro_auc = roc_auc_score(
            self.y_true_bin, self.probabilities, multi_class="ovr", average="micro"
        )

        # 计算加权平均
        class_counts = np.sum(self.y_true_bin, axis=0)
        weighted_auc = np.sum(class_aurocs * class_counts) / np.sum(class_counts)

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            # 各类别AUROC
            for j in range(self.n_classes):
                try:
                    auc_score = roc_auc_score(
                        self.y_true_bin[idx, j], self.probabilities[idx, j]
                    )
                except ValueError:
                    auc_score = 0.5
                bootstrap_class[j, i] = auc_score

            # 宏观平均
            bootstrap_macro[i] = np.mean(bootstrap_class[:, i])

            # 微观平均
            try:
                bootstrap_micro[i] = roc_auc_score(
                    self.y_true_bin[idx],
                    self.probabilities[idx],
                    multi_class="ovr",
                    average="micro",
                )
            except ValueError:
                bootstrap_micro[i] = roc_auc_score(
                    self.y_true_bin[idx].ravel(), self.probabilities[idx].ravel()
                )

            # 加权平均
            sample_counts = np.sum(self.y_true_bin[idx], axis=0)
            bootstrap_weighted[i] = np.sum(
                bootstrap_class[:, i] * sample_counts
            ) / np.sum(sample_counts)

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "AUROC": class_aurocs[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._auroc_full = {
            "per_class": class_results,
            "macro": {"AUROC": macro_auc, "CI": self._calculate_ci(bootstrap_macro)},
            "micro": {"AUROC": micro_auc, "CI": self._calculate_ci(bootstrap_micro)},
            "weighted": {
                "AUROC": weighted_auc,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._auroc_full["per_class"]
        return self._auroc_full[avg]

    def calculate_auprc(self, avg=None):
        """计算AUPRC及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_auprc_full"):
            if avg is None:
                return self._auprc_full["per_class"]
            return self._auprc_full[avg]

        # 原始计算
        class_auprcs = []
        for i in range(self.n_classes):
            precision, recall, _ = precision_recall_curve(
                self.y_true_bin[:, i], self.probabilities[:, i]
            )
            auprc = auc(recall, precision)
            class_auprcs.append(auprc)

        macro_auprc = np.mean(class_auprcs)
        micro_auprc = average_precision_score(
            self.y_true_bin, self.probabilities, average="micro"
        )

        # 计算加权平均
        class_counts = np.sum(self.y_true_bin, axis=0)
        weighted_auprc = np.sum(class_auprcs * class_counts) / np.sum(class_counts)

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            # 各类别AUPRC
            for j in range(self.n_classes):
                try:
                    precision, recall, _ = precision_recall_curve(
                        self.y_true_bin[idx, j], self.probabilities[idx, j]
                    )
                    auprc = auc(recall, precision)
                except ValueError:
                    auprc = 0.0
                bootstrap_class[j, i] = auprc

            # 宏观平均
            bootstrap_macro[i] = np.mean(bootstrap_class[:, i])

            # 微观平均
            try:
                bootstrap_micro[i] = average_precision_score(
                    self.y_true_bin[idx], self.probabilities[idx], average="micro"
                )
            except ValueError:
                bootstrap_micro[i] = 0.0

            # 加权平均
            sample_counts = np.sum(self.y_true_bin[idx], axis=0)
            bootstrap_weighted[i] = np.sum(
                bootstrap_class[:, i] * sample_counts
            ) / np.sum(sample_counts)

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "AUPRC": class_auprcs[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._auprc_full = {
            "per_class": class_results,
            "macro": {"AUPRC": macro_auprc, "CI": self._calculate_ci(bootstrap_macro)},
            "micro": {"AUPRC": micro_auprc, "CI": self._calculate_ci(bootstrap_micro)},
            "weighted": {
                "AUPRC": weighted_auprc,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._auprc_full["per_class"]
        return self._auprc_full[avg]

    def calculate_accuracy(self, avg=None):
        """计算准确率及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_accuracy_full"):
            if avg is None:
                return self._accuracy_full["per_class"]
            return self._accuracy_full[avg]

        # 计算混淆矩阵
        cm = confusion_matrix(self.ground_truth, self.y_pred, labels=self.classes)

        # 原始计算
        class_accuracies = []
        for i in range(self.n_classes):
            tp = cm[i, i]
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
            accuracy = (tp + tn) / self.n_samples
            class_accuracies.append(accuracy)

        # 计算各种平均
        macro_accuracy = np.mean(class_accuracies)
        micro_accuracy = accuracy_score(self.ground_truth, self.y_pred)

        class_counts = np.sum(cm, axis=1)
        weighted_accuracy = np.sum(np.array(class_accuracies) * class_counts) / np.sum(
            class_counts
        )

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            cm_bs = confusion_matrix(
                self.ground_truth[idx], self.y_pred[idx], labels=self.classes
            )

            # 各类别准确率
            class_acc_bs = []
            for j in range(self.n_classes):
                tp = cm_bs[j, j]
                tn = np.sum(cm_bs) - np.sum(cm_bs[j, :]) - np.sum(cm_bs[:, j]) + tp
                acc = (tp + tn) / len(idx)
                class_acc_bs.append(acc)
                bootstrap_class[j, i] = acc

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_acc_bs)

            # 微观平均
            bootstrap_micro[i] = accuracy_score(
                self.ground_truth[idx], self.y_pred[idx]
            )

            # 加权平均
            class_counts_bs = np.sum(cm_bs, axis=1)
            bootstrap_weighted[i] = np.sum(
                np.array(class_acc_bs) * class_counts_bs
            ) / np.sum(class_counts_bs)

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "Accuracy": class_accuracies[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._accuracy_full = {
            "per_class": class_results,
            "macro": {
                "Accuracy": macro_accuracy,
                "CI": self._calculate_ci(bootstrap_macro),
            },
            "micro": {
                "Accuracy": micro_accuracy,
                "CI": self._calculate_ci(bootstrap_micro),
            },
            "weighted": {
                "Accuracy": weighted_accuracy,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._accuracy_full["per_class"]
        return self._accuracy_full[avg]

    def calculate_sensitivity(self, avg=None):
        """计算敏感性(召回率)及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_sensitivity_full"):
            if avg is None:
                return self._sensitivity_full["per_class"]
            return self._sensitivity_full[avg]

        # 原始计算
        # 计算每个类别的敏感性
        class_sensitivities = recall_score(
            self.ground_truth, self.y_pred, average=None, zero_division=0
        )

        # 计算各种平均
        macro_sensitivity = np.mean(class_sensitivities)
        micro_sensitivity = recall_score(
            self.ground_truth, self.y_pred, average="micro", zero_division=0
        )
        weighted_sensitivity = recall_score(
            self.ground_truth, self.y_pred, average="weighted", zero_division=0
        )

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            # 各类别敏感性
            class_sens_bs = recall_score(
                self.ground_truth[idx], self.y_pred[idx], average=None, zero_division=0
            )
            for j in range(self.n_classes):
                bootstrap_class[j, i] = class_sens_bs[j]

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_sens_bs)

            # 微观平均
            bootstrap_micro[i] = recall_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="micro",
                zero_division=0,
            )

            # 加权平均
            bootstrap_weighted[i] = recall_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="weighted",
                zero_division=0,
            )

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "Sensitivity": class_sensitivities[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._sensitivity_full = {
            "per_class": class_results,
            "macro": {
                "Sensitivity": macro_sensitivity,
                "CI": self._calculate_ci(bootstrap_macro),
            },
            "micro": {
                "Sensitivity": micro_sensitivity,
                "CI": self._calculate_ci(bootstrap_micro),
            },
            "weighted": {
                "Sensitivity": weighted_sensitivity,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._sensitivity_full["per_class"]
        return self._sensitivity_full[avg]

    def calculate_specificity(self, avg=None):
        """计算特异性及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_specificity_full"):
            if avg is None:
                return self._specificity_full["per_class"]
            return self._specificity_full[avg]

        # 计算混淆矩阵
        cm = confusion_matrix(self.ground_truth, self.y_pred, labels=self.classes)

        # 原始计算
        class_specificities = []
        total_tn = 0
        total_fp = 0

        for i in range(self.n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)

            total_tn += tn
            total_fp += fp

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            class_specificities.append(specificity)

        # 计算各种平均
        macro_specificity = np.mean(class_specificities)
        micro_specificity = (
            total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
        )

        class_counts = np.sum(cm, axis=1)
        weighted_specificity = np.sum(
            np.array(class_specificities) * class_counts
        ) / np.sum(class_counts)

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            cm_bs = confusion_matrix(
                self.ground_truth[idx], self.y_pred[idx], labels=self.classes
            )

            total_tn_bs = 0
            total_fp_bs = 0
            class_spec_bs = []

            for j in range(self.n_classes):
                tp = cm_bs[j, j]
                fn = np.sum(cm_bs[j, :]) - tp
                fp = np.sum(cm_bs[:, j]) - tp
                tn = np.sum(cm_bs) - (tp + fn + fp)

                total_tn_bs += tn
                total_fp_bs += fp

                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                class_spec_bs.append(spec)
                bootstrap_class[j, i] = spec

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_spec_bs)

            # 微观平均
            bootstrap_micro[i] = (
                total_tn_bs / (total_tn_bs + total_fp_bs)
                if (total_tn_bs + total_fp_bs) > 0
                else 0.0
            )

            # 加权平均
            class_counts_bs = np.sum(cm_bs, axis=1)
            bootstrap_weighted[i] = np.sum(
                np.array(class_spec_bs) * class_counts_bs
            ) / np.sum(class_counts_bs)

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "Specificity": class_specificities[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._specificity_full = {
            "per_class": class_results,
            "macro": {
                "Specificity": macro_specificity,
                "CI": self._calculate_ci(bootstrap_macro),
            },
            "micro": {
                "Specificity": micro_specificity,
                "CI": self._calculate_ci(bootstrap_micro),
            },
            "weighted": {
                "Specificity": weighted_specificity,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._specificity_full["per_class"]
        return self._specificity_full[avg]

    def calculate_f1(self, avg=None):
        """计算F1分数及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_f1_full"):
            if avg is None:
                return self._f1_full["per_class"]
            return self._f1_full[avg]

        # 原始计算
        # 计算每个类别的F1分数
        class_f1s = f1_score(
            self.ground_truth, self.y_pred, average=None, zero_division=0
        )

        # 计算各种平均
        macro_f1 = np.mean(class_f1s)
        micro_f1 = f1_score(
            self.ground_truth, self.y_pred, average="micro", zero_division=0
        )
        weighted_f1 = f1_score(
            self.ground_truth, self.y_pred, average="weighted", zero_division=0
        )

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            # 各类别F1
            class_f1_bs = f1_score(
                self.ground_truth[idx], self.y_pred[idx], average=None, zero_division=0
            )
            for j in range(self.n_classes):
                bootstrap_class[j, i] = class_f1_bs[j]

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_f1_bs)

            # 微观平均
            bootstrap_micro[i] = f1_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="micro",
                zero_division=0,
            )

            # 加权平均
            bootstrap_weighted[i] = f1_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="weighted",
                zero_division=0,
            )

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "F1": class_f1s[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._f1_full = {
            "per_class": class_results,
            "macro": {"F1": macro_f1, "CI": self._calculate_ci(bootstrap_macro)},
            "micro": {"F1": micro_f1, "CI": self._calculate_ci(bootstrap_micro)},
            "weighted": {
                "F1": weighted_f1,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._f1_full["per_class"]
        return self._f1_full[avg]

    def calculate_ppv(self, avg=None):
        """计算阳性预测值(PPV/Precision)及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_ppv_full"):
            if avg is None:
                return self._ppv_full["per_class"]
            return self._ppv_full[avg]

        # 原始计算
        # 计算每个类别的PPV
        class_ppvs = precision_score(
            self.ground_truth, self.y_pred, average=None, zero_division=0
        )

        # 计算各种平均
        macro_ppv = np.mean(class_ppvs)
        micro_ppv = precision_score(
            self.ground_truth, self.y_pred, average="micro", zero_division=0
        )
        weighted_ppv = precision_score(
            self.ground_truth, self.y_pred, average="weighted", zero_division=0
        )

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            # 各类别PPV
            class_ppv_bs = precision_score(
                self.ground_truth[idx], self.y_pred[idx], average=None, zero_division=0
            )
            for j in range(self.n_classes):
                bootstrap_class[j, i] = class_ppv_bs[j]

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_ppv_bs)

            # 微观平均
            bootstrap_micro[i] = precision_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="micro",
                zero_division=0,
            )

            # 加权平均
            bootstrap_weighted[i] = precision_score(
                self.ground_truth[idx],
                self.y_pred[idx],
                average="weighted",
                zero_division=0,
            )

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "PPV": class_ppvs[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._ppv_full = {
            "per_class": class_results,
            "macro": {"PPV": macro_ppv, "CI": self._calculate_ci(bootstrap_macro)},
            "micro": {"PPV": micro_ppv, "CI": self._calculate_ci(bootstrap_micro)},
            "weighted": {
                "PPV": weighted_ppv,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._ppv_full["per_class"]
        return self._ppv_full[avg]

    def calculate_npv(self, avg=None):
        """计算阴性预测值(NPV)及95%置信区间

        参数:
        avg (str): 平均方式 (None, 'macro', 'micro', 'weighted')
                    None返回每个类别的结果
                    其他返回对应的平均结果
        """
        # 如果已经计算过全部结果，直接返回缓存
        if hasattr(self, "_npv_full"):
            if avg is None:
                return self._npv_full["per_class"]
            return self._npv_full[avg]

        # 计算混淆矩阵
        cm = confusion_matrix(self.ground_truth, self.y_pred, labels=self.classes)

        # 原始计算
        class_npvs = []
        total_tn = 0
        total_fn = 0

        for i in range(self.n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)

            total_tn += tn
            total_fn += fn

            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            class_npvs.append(npv)

        # 计算各种平均
        macro_npv = np.mean(class_npvs)
        micro_npv = (
            total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0.0
        )

        class_counts = np.sum(cm, axis=1)
        weighted_npv = np.sum(np.array(class_npvs) * class_counts) / np.sum(
            class_counts
        )

        # Bootstrap
        bootstrap_class = np.zeros((self.n_classes, self.n_bootstraps))
        bootstrap_macro = np.zeros(self.n_bootstraps)
        bootstrap_micro = np.zeros(self.n_bootstraps)
        bootstrap_weighted = np.zeros(self.n_bootstraps)

        for i, idx in enumerate(self.bootstrap_indices):
            cm_bs = confusion_matrix(
                self.ground_truth[idx], self.y_pred[idx], labels=self.classes
            )

            total_tn_bs = 0
            total_fn_bs = 0
            class_npv_bs = []

            for j in range(self.n_classes):
                tp = cm_bs[j, j]
                fn_bs = np.sum(cm_bs[j, :]) - tp
                fp = np.sum(cm_bs[:, j]) - tp
                tn_bs = np.sum(cm_bs) - (tp + fn_bs + fp)

                total_tn_bs += tn_bs
                total_fn_bs += fn_bs

                npv_bs = tn_bs / (tn_bs + fn_bs) if (tn_bs + fn_bs) > 0 else 0.0
                class_npv_bs.append(npv_bs)
                bootstrap_class[j, i] = npv_bs

            # 宏观平均
            bootstrap_macro[i] = np.mean(class_npv_bs)

            # 微观平均
            bootstrap_micro[i] = (
                total_tn_bs / (total_tn_bs + total_fn_bs)
                if (total_tn_bs + total_fn_bs) > 0
                else 0.0
            )

            # 加权平均
            class_counts_bs = np.sum(cm_bs, axis=1)
            bootstrap_weighted[i] = np.sum(
                np.array(class_npv_bs) * class_counts_bs
            ) / np.sum(class_counts_bs)

        # 整理结果
        class_results = {}
        for j, cls in enumerate(self.classes):
            class_results[cls] = {
                "NPV": class_npvs[j],
                "CI": self._calculate_ci(bootstrap_class[j]),
            }

        # 缓存全部结果
        self._npv_full = {
            "per_class": class_results,
            "macro": {"NPV": macro_npv, "CI": self._calculate_ci(bootstrap_macro)},
            "micro": {"NPV": micro_npv, "CI": self._calculate_ci(bootstrap_micro)},
            "weighted": {
                "NPV": weighted_npv,
                "CI": self._calculate_ci(bootstrap_weighted),
            },
        }

        # 根据参数返回结果
        if avg is None:
            return self._npv_full["per_class"]
        return self._npv_full[avg]

    def calculate_kappa(self, weights=None):
        """计算Cohen's Kappa系数及95%置信区间

        参数:
        weights (str): 权重类型 (None, 'linear', 'quadratic')
                    None: 不加权
                    'linear': 线性加权
                    'quadratic': 二次加权
        """
        # 生成缓存键名
        weights_key = "none" if weights is None else weights
        cache_attr = f"_kappa_{weights_key}_full"

        # 如果已经计算过，直接返回缓存
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        # 原始计算
        kappa = cohen_kappa_score(self.ground_truth, self.y_pred, weights=weights)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = cohen_kappa_score(
                self.ground_truth[idx], self.y_pred[idx], weights=weights
            )

        # 整理结果
        result = {
            "Kappa": kappa,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        # 缓存结果
        setattr(self, cache_attr, result)

        return result

    def calculate_mcc(self):
        """计算Matthews相关系数及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_mcc_full"):
            return self._mcc_full

        # 原始计算
        mcc = matthews_corrcoef(self.ground_truth, self.y_pred)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = matthews_corrcoef(
                self.ground_truth[idx], self.y_pred[idx]
            )

        # 整理结果
        self._mcc_full = {"MCC": {"MCC": mcc, "CI": self._calculate_ci(bootstrap_vals)}}

        return self._mcc_full

    def calculate_all_metrics(self):
        """计算所有指标并返回综合结果"""
        results = OrderedDict()

        # AUROC
        auroc_results = OrderedDict()
        auroc_results["per_class"] = self.calculate_auroc(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            auroc_results[avg] = self.calculate_auroc(avg=avg)
        results["AUROC"] = auroc_results

        # AUPRC
        auprc_results = OrderedDict()
        auprc_results["per_class"] = self.calculate_auprc(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            auprc_results[avg] = self.calculate_auprc(avg=avg)
        results["AUPRC"] = auprc_results

        # Accuracy
        accuracy_results = OrderedDict()
        accuracy_results["per_class"] = self.calculate_accuracy(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            accuracy_results[avg] = self.calculate_accuracy(avg=avg)
        results["Accuracy"] = accuracy_results

        # Sensitivity
        sensitivity_results = OrderedDict()
        sensitivity_results["per_class"] = self.calculate_sensitivity(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            sensitivity_results[avg] = self.calculate_sensitivity(avg=avg)
        results["Sensitivity"] = sensitivity_results

        # Specificity
        specificity_results = OrderedDict()
        specificity_results["per_class"] = self.calculate_specificity(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            specificity_results[avg] = self.calculate_specificity(avg=avg)
        results["Specificity"] = specificity_results

        # F1
        f1_results = OrderedDict()
        f1_results["per_class"] = self.calculate_f1(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            f1_results[avg] = self.calculate_f1(avg=avg)
        results["F1"] = f1_results

        # PPV
        ppv_results = OrderedDict()
        ppv_results["per_class"] = self.calculate_ppv(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            ppv_results[avg] = self.calculate_ppv(avg=avg)
        results["PPV"] = ppv_results

        # NPV
        npv_results = OrderedDict()
        npv_results["per_class"] = self.calculate_npv(avg=None)
        for avg in ["macro", "micro", "weighted"]:
            npv_results[avg] = self.calculate_npv(avg=avg)
        results["NPV"] = npv_results

        # Kappa
        kappa_results = OrderedDict()
        for weights in [None, "linear", "quadratic"]:
            weights_key = "none" if weights is None else weights
            kappa_results[f"Kappe_{weights_key}"] = self.calculate_kappa(
                weights=weights
            )
        results["Kappa"] = kappa_results

        # MCC
        results["MCC"] = self.calculate_mcc()

        return results


def parse_string_to_array(s):
    return np.fromstring(s.strip("[]"), sep=" ")


if __name__ == "__main__":
    csv_path = "/data_A/xujialiu/projects/0_personal/my_utils/calculate_metrics/probabilities_ground_truth/test_data.csv"
    output_excel_name = (
        "results.xlsx"
    )
    n_bootstraps = 100
    
    output_excel_path = Path(csv_path).parent / output_excel_name

    df = pd.read_csv(csv_path)
    ground_truth = df.ground_truths
    probabilities = df.probabilities.map(parse_string_to_array)

    results = MulticlassMetricsCalculator(
        ground_truth, probabilities, n_bootstraps, random_seed=0
    ).calculate_all_metrics()

    pprint(results)
    pd.DataFrame(results).to_excel(output_excel_path)
