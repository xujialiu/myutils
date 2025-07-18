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
from pprint import pprint
from pathlib import Path


class BinaryMetricsCalculator:
    def __init__(
        self, ground_truth, probabilities, n_bootstraps=1000, random_seed=None
    ):
        """
        初始化二分类指标计算器

        参数:
        ground_truth (pd.Series): 真实标签 (0或1)
        probabilities (pd.Series): 预测概率，正类(类别1)的概率
        n_bootstraps (int): Bootstrap采样次数，默认为1000
        random_seed (int): 随机种子，确保结果可复现
        """
        self.ground_truth = ground_truth.values
        self.probabilities = probabilities.values
        self.n_bootstraps = n_bootstraps
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

        # 生成预测标签（使用0.5作为阈值）
        self.y_pred = (self.probabilities >= 0.5).astype(int)

    def _calculate_ci(self, values):
        """计算95%置信区间"""
        alpha = 100 * 0.05 / 2
        return (np.percentile(values, alpha), np.percentile(values, 100 - alpha))

    def calculate_auroc(self):
        """计算AUROC及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_auroc_result"):
            return self._auroc_result

        # 原始计算
        auroc = roc_auc_score(self.ground_truth, self.probabilities)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            try:
                bootstrap_vals[i] = roc_auc_score(
                    self.ground_truth[idx], self.probabilities[idx]
                )
            except ValueError:
                bootstrap_vals[i] = 0.5

        # 缓存结果
        self._auroc_result = {
            "AUROC": auroc,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._auroc_result

    def calculate_auprc(self):
        """计算AUPRC及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_auprc_result"):
            return self._auprc_result

        # 原始计算
        precision, recall, _ = precision_recall_curve(
            self.ground_truth, self.probabilities
        )
        auprc = auc(recall, precision)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            try:
                precision_bs, recall_bs, _ = precision_recall_curve(
                    self.ground_truth[idx], self.probabilities[idx]
                )
                bootstrap_vals[i] = auc(recall_bs, precision_bs)
            except ValueError:
                bootstrap_vals[i] = 0.0

        # 缓存结果
        self._auprc_result = {
            "AUPRC": auprc,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._auprc_result

    def calculate_accuracy(self):
        """计算准确率及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_accuracy_result"):
            return self._accuracy_result

        # 原始计算
        accuracy = accuracy_score(self.ground_truth, self.y_pred)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = accuracy_score(self.ground_truth[idx], self.y_pred[idx])

        # 缓存结果
        self._accuracy_result = {
            "Accuracy": accuracy,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._accuracy_result

    def calculate_sensitivity(self):
        """计算敏感性(召回率)及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_sensitivity_result"):
            return self._sensitivity_result

        # 原始计算
        sensitivity = recall_score(self.ground_truth, self.y_pred, zero_division=0)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = recall_score(
                self.ground_truth[idx], self.y_pred[idx], zero_division=0
            )

        # 缓存结果
        self._sensitivity_result = {
            "Sensitivity": sensitivity,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._sensitivity_result

    def calculate_specificity(self):
        """计算特异性及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_specificity_result"):
            return self._specificity_result

        # 计算混淆矩阵
        cm = confusion_matrix(self.ground_truth, self.y_pred)

        # 原始计算
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            cm_bs = confusion_matrix(self.ground_truth[idx], self.y_pred[idx])
            tn_bs = cm_bs[0, 0]
            fp_bs = cm_bs[0, 1]
            bootstrap_vals[i] = tn_bs / (tn_bs + fp_bs) if (tn_bs + fp_bs) > 0 else 0.0

        # 缓存结果
        self._specificity_result = {
            "Specificity": specificity,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._specificity_result

    def calculate_f1(self):
        """计算F1分数及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_f1_result"):
            return self._f1_result

        # 原始计算
        f1 = f1_score(self.ground_truth, self.y_pred, zero_division=0)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = f1_score(
                self.ground_truth[idx], self.y_pred[idx], zero_division=0
            )

        # 缓存结果
        self._f1_result = {
            "F1": f1,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._f1_result

    def calculate_ppv(self):
        """计算阳性预测值(PPV/Precision)及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_ppv_result"):
            return self._ppv_result

        # 原始计算
        ppv = precision_score(self.ground_truth, self.y_pred, zero_division=0)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = precision_score(
                self.ground_truth[idx], self.y_pred[idx], zero_division=0
            )

        # 缓存结果
        self._ppv_result = {
            "PPV": ppv,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._ppv_result

    def calculate_npv(self):
        """计算阴性预测值(NPV)及95%置信区间"""
        # 如果已经计算过，直接返回缓存
        if hasattr(self, "_npv_result"):
            return self._npv_result

        # 计算混淆矩阵
        cm = confusion_matrix(self.ground_truth, self.y_pred)

        # 原始计算
        tn = cm[0, 0]
        fn = cm[1, 0]
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            cm_bs = confusion_matrix(self.ground_truth[idx], self.y_pred[idx])
            tn_bs = cm_bs[0, 0]
            fn_bs = cm_bs[1, 0]
            bootstrap_vals[i] = tn_bs / (tn_bs + fn_bs) if (tn_bs + fn_bs) > 0 else 0.0

        # 缓存结果
        self._npv_result = {
            "NPV": npv,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._npv_result

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
        cache_attr = f"_kappa_{weights_key}_result"

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
        if hasattr(self, "_mcc_result"):
            return self._mcc_result

        # 原始计算
        mcc = matthews_corrcoef(self.ground_truth, self.y_pred)

        # Bootstrap
        bootstrap_vals = np.zeros(self.n_bootstraps)
        for i, idx in enumerate(self.bootstrap_indices):
            bootstrap_vals[i] = matthews_corrcoef(
                self.ground_truth[idx], self.y_pred[idx]
            )

        # 缓存结果
        self._mcc_result = {
            "MCC": mcc,
            "CI": self._calculate_ci(bootstrap_vals),
        }

        return self._mcc_result

    def calculate_all_metrics(self):
        """计算所有指标并返回综合结果"""
        results = OrderedDict()

        # AUROC
        results["AUROC"] = self.calculate_auroc()

        # AUPRC
        results["AUPRC"] = self.calculate_auprc()

        # Accuracy
        results["Accuracy"] = self.calculate_accuracy()

        # Sensitivity (Recall)
        results["Sensitivity"] = self.calculate_sensitivity()

        # Specificity
        results["Specificity"] = self.calculate_specificity()

        # F1
        results["F1"] = self.calculate_f1()

        # PPV (Precision)
        results["PPV"] = self.calculate_ppv()

        # NPV
        results["NPV"] = self.calculate_npv()

        # Kappa
        kappa_results = OrderedDict()
        for weights in [None, "linear", "quadratic"]:
            weights_key = "none" if weights is None else weights
            kappa_results[f"Kappa_{weights_key}"] = self.calculate_kappa(
                weights=weights
            )
        results["Kappa"] = kappa_results

        # MCC
        results["MCC"] = self.calculate_mcc()

        return results

    def get_optimal_threshold(self, metric="youden"):
        """
        计算最优阈值

        参数:
        metric (str): 优化的指标
                     'youden': Youden's J statistic (Sensitivity + Specificity - 1)
                     'f1': F1 score
                     'accuracy': Accuracy
        """
        thresholds = np.unique(self.probabilities)
        thresholds = np.sort(thresholds)

        best_threshold = 0.5
        best_score = -np.inf

        for threshold in thresholds:
            y_pred_temp = (self.probabilities >= threshold).astype(int)

            if metric == "youden":
                cm = confusion_matrix(self.ground_truth, y_pred_temp)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == "f1":
                score = f1_score(self.ground_truth, y_pred_temp, zero_division=0)
            elif metric == "accuracy":
                score = accuracy_score(self.ground_truth, y_pred_temp)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return {"threshold": best_threshold, "score": best_score, "metric": metric}


def format_results_for_excel(results):
    """将结果格式化为适合Excel输出的格式"""
    formatted_data = []

    for metric_name, metric_value in results.items():
        if metric_name == "Kappa":
            # Kappa有多个子项
            for kappa_type, kappa_value in metric_value.items():
                row = {
                    "Metric": kappa_type,
                    "Value": kappa_value["Kappa"],
                    "CI_Lower": kappa_value["CI"][0],
                    "CI_Upper": kappa_value["CI"][1],
                    "CI_95%": f"({kappa_value['CI'][0]:.4f}, {kappa_value['CI'][1]:.4f})",
                }
                formatted_data.append(row)
        else:
            # 其他指标
            row = {
                "Metric": metric_name,
                "Value": metric_value[metric_name],
                "CI_Lower": metric_value["CI"][0],
                "CI_Upper": metric_value["CI"][1],
                "CI_95%": f"({metric_value['CI'][0]:.4f}, {metric_value['CI'][1]:.4f})",
            }
            formatted_data.append(row)

    return pd.DataFrame(formatted_data)


def parse_string_to_array(s):
    return np.fromstring(s.strip("[]"), sep=" ")[1]


if __name__ == "__main__":
    # 示例用法
    csv_path = "/data_A/xujialiu/projects/0_personal/my_utils/calculate_metrics/probabilities_ground_truth/test_data_binary.csv"
    output_excel_name = "binary_results.xlsx"
    n_bootstraps = 100

    output_excel_path = Path(csv_path).parent / output_excel_name

    # 读取数据
    df = pd.read_csv(csv_path)
    ground_truth = df.ground_truths  # 0 或 1
    probabilities = df.probabilities.map(parse_string_to_array)  # 正类的概率

    # 初始化计算器
    calculator = BinaryMetricsCalculator(
        ground_truth, probabilities, n_bootstraps, random_seed=0
    )

    # 计算所有指标
    results = calculator.calculate_all_metrics()

    # 打印结果
    pprint(results)

    # 计算最优阈值
    optimal_threshold_youden = calculator.get_optimal_threshold(metric="youden")
    optimal_threshold_f1 = calculator.get_optimal_threshold(metric="f1")

    print("\n最优阈值:")
    print(f"Youden's J: {optimal_threshold_youden}")
    print(f"F1 Score: {optimal_threshold_f1}")

    # 保存到Excel
    df_results = format_results_for_excel(results)

    # 添加最优阈值信息
    threshold_info = pd.DataFrame(
        [
            {
                "Metric": "Optimal_Threshold_Youden",
                "Value": optimal_threshold_youden["threshold"],
                "CI_Lower": optimal_threshold_youden["score"],
                "CI_Upper": optimal_threshold_youden["score"],
                "CI_95%": f"Score: {optimal_threshold_youden['score']:.4f}",
            },
            {
                "Metric": "Optimal_Threshold_F1",
                "Value": optimal_threshold_f1["threshold"],
                "CI_Lower": optimal_threshold_f1["score"],
                "CI_Upper": optimal_threshold_f1["score"],
                "CI_95%": f"Score: {optimal_threshold_f1['score']:.4f}",
            },
        ]
    )

    df_results = pd.concat([df_results, threshold_info], ignore_index=True)
    df_results.to_excel(output_excel_path, index=False)
    print(f"\n结果已保存到: {output_excel_path}")
