import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class fisheriris:
    def __init__(self, data):
        mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
        self.data = data.copy()
        self.data["Class"] = self.data["Class"].str.lower().map(mapping).astype(int)
        self.display_info()
        self.basic_stats()  # Basic stats calculated before normalization
        self.within_variance()  # Within-class variance before normalization
        self.between_variance()  # Between-class variance before normalization
        self.plot_corr()  # Correlation plot before normalization
        self.plot_feat_vs_class()  # Feature vs Class plots before normalization
        self.normalize_data()

    def normalize_data(self):
        cols = ["SepL", "SepW", "PetL", "PetW"]
        for col in cols:
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std(ddof=0)

    def display_info(self):
        ncls = len(self.data["Class"].unique())
        nfeat = self.data.shape[1] - 1
        print("Dataset Information:")
        print("  Number of classes:", ncls)
        print("  Number of features:", nfeat)
        print("  Iris measurements used for species classification.\n")

    def basic_stats(self):
        print("=== Basic Statistics ===")
        print(self.data.iloc[:, :-1].agg(['min', 'max', 'mean', 'var']))

    def within_variance(self):
        cls = self.data["Class"].unique()
        print("\n=== Within-Class Variance ===")
        res = {}
        for feat in self.data.columns[:-1]:
            s = 0
            for c in cls:
                sub = self.data[self.data["Class"] == c][feat]
                p = len(sub) / len(self.data)
                s += p * np.var(sub, ddof=0)
            res[feat] = s
        print(res)

    def between_variance(self):
        overall = self.data.iloc[:, :-1].mean()
        cls = self.data["Class"].unique()
        print("\n=== Between-Class Variance ===")
        res = {}
        for feat in self.data.columns[:-1]:
            s = 0
            for c in cls:
                sub = self.data[self.data["Class"] == c][feat]
                p = len(sub) / len(self.data)
                mu = np.mean(sub)
                s += p * (mu - overall[feat])**2
            res[feat] = s
        print(res)

    def corr_matrix(self):
        return self.data.corr()

    def plot_corr(self):
        plt.figure(figsize=(6, 5))
        sns.heatmap(self.corr_matrix(), annot=False, cmap="jet", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feat_vs_class(self):
        feats = ["SepL", "SepW", "PetL", "PetW"]
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.ravel()
        for i in range(4):
            axs[i].scatter(self.data[feats[i]], self.data["Class"] + 1, color='red', s=10, marker='x')
            axs[i].set_xlabel(feats[i])
            axs[i].set_ylabel("Class")
            axs[i].set_title(f"{feats[i]} vs Class")
        plt.tight_layout()
        plt.show()

    def prep_binary(self, pos_class, feats):
        d = self.data.copy()
        labels = np.where(d["Class"] == pos_class, 1, -1)
        X = d[feats].values
        X = np.c_[np.ones(X.shape[0]), X]
        return X, labels

    def perceptron(self, X, labels, max_epochs=1000):
        w = np.zeros(X.shape[1])
        ep = 0
        conv = False
        while ep < max_epochs:
            pred = np.sign(X @ w)
            mis = (pred != labels)
            if not mis.any():
                conv = True
                break
            w += (X[mis].T @ labels[mis]) / np.sum(mis)
            ep += 1
        return w, ep, int(np.sum(mis)), conv

    def ls_solution(self, X, labels):
        return np.linalg.pinv(X) @ labels

    def multiclass_ls(self, feats):
        X = self.data[feats].values
        X = np.c_[np.ones(X.shape[0]), X]
        onehot = np.eye(3)[self.data["Class"].values]
        return np.linalg.pinv(X) @ onehot

    def plot_boundary(self, X, labels, w, title):
        plt.figure(figsize=(6, 5))
        plt.scatter(X[labels == 1, 1], X[labels == 1, 2], color='blue', label='Positive (+1)')
        plt.scatter(X[labels == -1, 1], X[labels == -1, 2], color='red', label='Negative (-1)')
        x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = np.sign(w[0] + w[1] * xx + w[2] * yy)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.show()

    def run_tasks(self):
        tasks = [
            ("Setosa vs Versicolor+Virginica (All Features)", 0, ["SepL", "SepW", "PetL", "PetW"]),
            ("Setosa vs Versicolor+Virginica (Features 3 & 4)", 0, ["PetL", "PetW"]),
            ("Virginica vs Versicolor+Setosa (All Features)", 2, ["SepL", "SepW", "PetL", "PetW"]),
            ("Virginica vs Versicolor+Setosa (Features 3 & 4)", 2, ["PetL", "PetW"])
        ]
        for desc, pos, feats in tasks:
            print("\n====", desc, "====")
            X, labels = self.prep_binary(pos, feats)
            w_p, ep, mis, conv = self.perceptron(X, labels)
            print("Perceptron converged:", conv)
            print("Epochs:", ep)
            print("Perceptron weights:", w_p)
            print("Misclassifications:", mis)
            w_ls = self.ls_solution(X, labels)
            print("LS weights:", w_ls)
            if len(feats) == 2:
                self.plot_boundary(X, labels, w_p, "" + desc)
        print("\n==== Setosa vs Versicolor vs Virginica (Multiclass LS, Features 3 & 4) ====")
        W_multi = self.multiclass_ls(["PetL", "PetW"])
        print("Multiclass LS Weight Matrix:")
        print(W_multi)

if __name__ == "__main__":
    file_path = "Proj1DataSet.xlsx"
    data = pd.read_excel(file_path)
    data.columns = ["SepL", "SepW", "PetL", "PetW", "Class"]
    iris = fisheriris(data)
    iris.run_tasks()
