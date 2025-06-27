import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from datetime import datetime
import os

class Data:
    """数据生成与处理类，用于生成不同类型的二维分类数据集并进行划分。"""

    def generate_linearly_separable_data(self, seed=1):
        """
        生成二维线性可分数据集。
        
        参数:
            seed (int): 随机种子，确保实验可重复性，默认为1。
        
        返回:
            tuple: (X1, y1, X2, y2)
                - X1 (ndarray): 正类样本数据，形状为(100, 2)。
                - y1 (ndarray): 正类标签，全为+1，形状为(100,)。
                - X2 (ndarray): 负类样本数据，形状为(100, 2)。
                - y2 (ndarray): 负类标签，全为-1，形状为(100,)。
        """
        np.random.seed(seed)
        mean1 = np.array([0, 2])  
        mean2 = np.array([2, 0]) 
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])  
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_non_linearly_separable_data(self, seed=1):
        """
        生成二维非线性可分数据集。
        
        参数:
            seed (int): 随机种子，默认为1。
        
        返回:
            tuple: (X1, y1, X2, y2)
                - X1 (ndarray): 正类样本数据，形状为(100, 2)，包含两个簇。
                - y1 (ndarray): 正类标签，全为+1，形状为(100,)。
                - X2 (ndarray): 负类样本数据，形状为(100, 2)，包含两个簇。
                - y2 (ndarray): 负类标签，全为-1，形状为(100,)。
        """
        np.random.seed(seed)
        mean1, mean3 = [-1, 2], [4, -4]  
        mean2, mean4 = [1, -1], [-4, 4]  
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])  
        X1 = np.vstack([
            np.random.multivariate_normal(mean1, cov, 50),
            np.random.multivariate_normal(mean3, cov, 50)
        ])
        y1 = np.ones(len(X1))
        X2 = np.vstack([
            np.random.multivariate_normal(mean2, cov, 50),
            np.random.multivariate_normal(mean4, cov, 50)
        ])
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_linearly_separable_overlap_data(self, seed=1):
        """
        生成二维线性可分但有部分重叠的数据集。
        
        参数:
            seed (int): 随机种子，默认为1。
        
        返回:
            tuple: (X1, y1, X2, y2)
                - X1 (ndarray): 正类样本数据，形状为(100, 2)。
                - y1 (ndarray): 正类标签，全为+1，形状为(100,)。
                - X2 (ndarray): 负类样本数据，形状为(100, 2)。
                - y2 (ndarray): 负类标签，全为-1，形状为(100,)。
        """
        np.random.seed(seed)
        mean1 = np.array([0, 2])  
        mean2 = np.array([2, 0])  
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])  
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_data(self, X1, y1, X2, y2, test_ratio=0.2):
        """
        将数据集划分为训练集和测试集。
        
        参数:
            X1 (ndarray): 正类样本数据。
            y1 (ndarray): 正类标签。
            X2 (ndarray): 负类样本数据。
            y2 (ndarray): 负类标签。
            test_ratio (float): 测试集比例，范围为(0, 1)，默认为0.2。
        
        返回:
            tuple: (X_train, y_train, X_test, y_test)
                - X_train (ndarray): 训练集特征，形状为(2*cutoff, 2)。
                - y_train (ndarray): 训练集标签，形状为(2*cutoff,)。
                - X_test (ndarray): 测试集特征，形状为(2*(N-cutoff), 2)。
                - y_test (ndarray): 测试集标签，形状为(2*(N-cutoff),)。
        
        异常:
            Value-DRIVER: 如果test_ratio不在(0, 1)范围内。
        """
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
        N = len(X1)
        cutoff = int(N * (1 - test_ratio)) 
        
        X_train = np.vstack((X1[:cutoff], X2[:cutoff]))
        y_train = np.hstack((y1[:cutoff], y2[:cutoff]))
        
        X_test = np.vstack((X1[cutoff:], X2[cutoff:]))
        y_test = np.hstack((y1[cutoff:], y2[cutoff:]))
        
        return X_train, y_train, X_test, y_test

class Plotting:
    """绘图类，用于可视化SVM分类结果和决策边界。"""

    def __init__(self, save_dir="./results"):
        """
        初始化绘图类，设置保存路径。
        
        参数:
            save_dir (str): 保存图像的目录，默认为"./results"。
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_margin(self, X_pos, X_neg, model, suffix=""):
        """
        绘制SVM分类结果，包括数据点、支持向量和决策边界。
        
        参数:
            X_pos (ndarray): 正类样本数据，形状为(N, 2)。
            X_neg (ndarray): 负类样本数据，形状为(N, 2)。
            model: SVM模型对象，需包含fit_type ('custom'或'sklearn')和相关属性（如w, b, sv等）。
            suffix (str): 保存图像的文件名后缀，默认为空。
        
        行为:
            - 绘制正类（X）和负类（O）数据点。
            - 标记支持向量（蓝色点）。
            - 绘制决策边界（实线）和间隔边界（虚线）。
            - 保存图像到self.save_dir，文件名包含时间戳和suffix。
        """
        def decision_line(x, w, b, c=0):
            """计算决策边界或间隔边界的x2值：w·x + b = c"""
            return (-w[0] * x - b + c) / w[1]

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.set_facecolor('#E6E6E6')  
        ax.set_axisbelow(True)
        plt.grid(color='w', linestyle='solid')  
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_color('gray')
        
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=8)
        
        x1_min, x1_max = min(X_pos[:, 0].min(), X_neg[:, 0].min()), max(X_pos[:, 0].max(), X_neg[:, 0].max())
        x2_min, x2_max = min(X_pos[:, 1].min(), X_neg[:, 1].min()), max(X_pos[:, 1].max(), X_neg[:, 1].max())
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        plt.title(f'Support Vector Machine - Library: {model.fit_type} - Kernel: {model.kernel}', fontsize=9)
        
        legend_elements = [
            Line2D([0], [0], linestyle='none', marker='x', color='lightblue', markersize=9),
            Line2D([0], [0], linestyle='none', marker='o', color='darkorange', markersize=9),
            Line2D([0], [0], linestyle='-', color='black', markersize=0),
            Line2D([0], [0], linestyle='--', color='black', markersize=0),
            Line2D([0], [0], linestyle='none', marker='.', color='blue', markersize=9)
        ]
        legend_loc = 'lower left' if model.kernel != 'linear' else 'upper left'
        legend_bbox = (0.03, 0.03) if model.kernel != 'linear' else (0.3, 0.98)
        plt.legend(legend_elements, ['Negative Class -1', 'Positive Class +1', 'Decision Boundary', 'Margin', 'Support Vectors'],
                   fontsize=7, shadow=True, loc=legend_loc, bbox_to_anchor=legend_bbox)
        
        plt.plot(X_pos[:, 0], X_pos[:, 1], 'x', markersize=5, color='lightblue')
        plt.plot(X_neg[:, 0], X_neg[:, 1], 'o', markersize=4, color='darkorange')
        
        sv = model.support_vectors_ if model.fit_type == 'sklearn' else model.sv
        plt.scatter(sv[:, 0], sv[:, 1], s=60, color='blue')
        
        if model.fit_type == 'sklearn' or model.kernel in ['polynomial', 'gaussian']:
            X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50))
            X_grid = np.c_[X1.ravel(), X2.ravel()]
            Z = (model.decision_function(X_grid) if model.fit_type == 'sklearn' else model.project(X_grid)).reshape(X1.shape)
            plt.contour(X1, X2, Z, levels=[0], colors='k', linewidths=1)
            plt.contour(X1, X2, Z, levels=[-1, 1], colors='grey', linestyles='--', linewidths=1)
        else:
            x = np.array([x1_min, x1_max])
            plt.plot(x, decision_line(x, model.w, model.b), 'k-')  
            plt.plot(x, decision_line(x, model.w, model.b, 1), 'k--')  
            plt.plot(x, decision_line(x, model.w, model.b, -1), 'k--')  
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/svm_{suffix}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

class SVM:
    """支持向量机类，实现自定义SVM算法，支持线性核、多项式核和高斯核。"""

    def __init__(self, kernel='linear', C=0.0, gamma=1.0, degree=3):
        """
        初始化SVM模型。
        
        参数:
            kernel (str): 核函数类型，支持'linear'、'polynomial'、'gaussian'，默认为'linear'。
            C (float): 软间隔惩罚参数，控制误分类容忍度，默认为0.0（硬间隔）。
            gamma (float): 高斯核参数，控制径向基函数宽度，默认为1.0。必须为正浮点数。
            degree (int): 多项式核的阶数，默认为3。
        
        异常:
            ValueError: 如果kernel不在支持的类型中或gamma不是正浮点数。
        """
        # TODO: 初始化SVM模型
        # 1. 验证kernel是否在['linear', 'polynomial', 'gaussian']中，若不在抛出ValueError。
        # 2. 将C转换为浮点数，若为None则设为0.0。
        # 3. 验证gamma是否为正浮点数，若不是抛出ValueError（可尝试转换为float并检查>0）。
        # 4. 将degree转换为整数，若为None则设为3。
        # 5. 初始化以下实例变量：
        #    - self.kernel: 核函数类型（str）
        #    - self.C: 软间隔参数（float）
        #    - self.gamma: 高斯核参数（float）
        #    - self.degree: 多项式核阶数（int）
        #    - self.w: 权重向量（ndarray，仅线性核使用，初始为None）
        #    - self.b: 偏置项（float，初始为None）
        #    - self.alphas: 拉格朗日乘子（ndarray，初始为None）
        #    - self.sv: 支持向量（ndarray，初始为None）
        #    - self.sv_y: 支持向量标签（ndarray，初始为None）
        #    - self.fit_type: 模型类型，设为'custom'
        pass

    def _kernel_function(self, x1, x2):
        """
        计算核函数值。
        
        参数:
            x1 (ndarray): 第一个输入向量，形状为(n_features,)。
            x2 (ndarray): 第二个输入向量，形状为(n_features,)。
        
        返回:
            float: 核函数值。
        """
        # TODO: 实现核函数计算
        # 1. 根据self.kernel的值选择核函数：
        #    - 'linear': 计算x1和x2的点积。
        #    - 'polynomial': 计算多项式核 (x1·x2 + 1)^degree，其中degree为self.degree。
        #    - 'gaussian': 计算高斯核 exp(-gamma * ||x1 - x2||^2)，其中gamma为self.gamma。
        # 2. 返回计算得到的核函数值（float）。
        # 注意：
        # - 使用NumPy进行向量运算（如np.dot、np.linalg.norm）。
        # - 确保输入x1, x2是形状相同的向量。
        pass

    def fit(self, X, y):
        """
        训练SVM模型，使用二次规划求解拉格朗日乘子。
        
        参数:
            X (ndarray): 训练数据特征，形状为(n_samples, n_features)。
            y (ndarray): 训练数据标签，形状为(n_samples,)，值为+1或-1。
        
        行为:
            - 计算核矩阵并使用cvxopt求解二次规划问题。
            - 提取支持向量及其拉格朗日乘子。
            - 计算偏置项b（对所有支持向量取平均）。
            - 对于线性核，计算权重向量w。
        """
        # TODO: 实现SVM训练
        # 1. 获取样本数n_samples和特征数n_features。
        # 2. 计算核矩阵K（n_samples x n_samples），其中K[i,j] = kernel_function(X[i], X[j])。
        # 3. 设置二次规划参数（使用cvxopt.matrix）：
        #    - P = y_i * y_j * K[i,j]（外积矩阵）。
        #    - q = -1向量（n_samples,）。
        #    - A = y（行向量，形状为(1, n_samples)）。
        #    - b = 0.0。
        #    - 如果self.C == 0（硬间隔）：
        #        - G = -I（单位矩阵的负值）。
        #        - h = 0向量。
        #    - 否则（软间隔）：
        #        - G = 垂直堆叠(-I, I)。
        #        - h = 水平堆叠(0向量, C向量)。
        # 4. 配置cvxopt求解器选项（关闭显示进度，设置高精度：abstol=1e-10, reltol=1e-10, feastol=1e-10）。
        # 5. 使用cvxopt.solvers.qp(P, q, G, h, A, b)求解，提取拉格朗日乘子alphas。
        # 6. 提取支持向量：
        #    - 选择alphas > 1e-5的索引。
        #    - 存储self.alphas（支持向量的拉格朗日乘子）。
        #    - 存储self.sv（支持向量，X的子集）。
        #    - 存储self.sv_y（支持向量标签，y的子集）。
        # 7. 计算偏置项self.b：
        #    - 对每个支持向量，计算b_i = y_i - sum(alpha_j * y_j * K[i,j])。
        #    - 取所有b_i的平均值。
        # 8. 如果是线性核（self.kernel == 'linear'）：
        #    - 计算权重self.w = sum(alpha_i * y_i * x_i)。
        #    - 否则，self.w = None。
        # 注意：
        # - 确保矩阵维度正确（如P为n_samples x n_samples，A为1 x n_samples）。
        # - 使用cvxopt.matrix确保数据类型为'd'（double）。
        # - 处理数值稳定性（如设置求解器精度）。
        pass

    def project(self, X):
        """
        计算决策函数值（未取符号）。
        
        参数:
            X (ndarray): 输入特征矩阵，形状为(n_samples, n_features)。
        
        返回:
            ndarray: 决策函数值，形状为(n_samples,)。
        """
        # TODO: 实现决策函数计算
        # 1. 如果self.w不为空（线性核）：
        #    - 计算f(x) = X·w + b，返回结果（形状为(n_samples,)）。
        # 2. 否则（非线性核）：
        #    - 初始化输出数组y_pred（形状为(n_samples,)）。
        #    - 对每个输入样本X[i]：
        #        - 计算f(x) = sum(alpha_j * y_j * kernel_function(X[i], sv_j)) + b。
        #        - 其中sv_j为支持向量（self.sv），alpha_j为self.alphas，y_j为self.sv_y。
        #    - 存储结果到y_pred[i]。
        # 3. 返回y_pred。
        # 注意：
        # - 使用NumPy进行向量运算以提高效率。
        # - 确保self.b、self.sv等已正确初始化。
        pass

    def predict(self, X):
        """
        预测样本类别。
        
        参数:
            X (ndarray): 输入特征矩阵，形状为(n_samples, n_features)。
        
        返回:
            ndarray: 预测标签，形状为(n_samples,)，值为+1或-1。
        """
        # TODO: 实现类别预测
        # 1. 调用self.project(X)获取决策函数值。
        # 2. 对决策函数值应用sign函数，返回+1（正）或-1（负）。
        # 3. 返回预测标签（形状为(n_samples,)）。
        # 注意：
        # - 使用np.sign确保输出为+1或-1。
        # - 处理决策函数值为0的情况（可归为+1或-1，按需选择）。
        pass

def formatted_output(model):
    """
    输出SVM训练和预测结果摘要。
    
    参数:
        model: SVM模型对象，需包含number_of_train_examples、correct_predictions等属性。
    
    行为:
        - 打印训练样本数、支持向量数、测试样本数和正确预测数。
    """
    dash = '=' * 60
    print(dash)
    print("        Support Vector Machine Results")
    print(dash)
    print(f"               **** In-Sample: ****")
    print(f"Found {len(model.sv)} support vectors from {model.number_of_train_examples} samples.")
    print(f"               **** Prediction: ****")
    print(f"{model.correct_predictions} correct predictions out of {model.number_of_test_examples}.")
    print(dash)

def run_custom_svm(kernel='linear', C=0.0, gamma=0.1, degree=3):
    """
    运行自定义SVM算法并评估结果。
    
    参数:
        kernel (str): 核函数类型，默认为'linear'。
        C (float): 软间隔参数，默认为0.0。
        gamma (float): 高斯核参数，必须为正浮点数，默认为0.1。
        degree (int): 多项式核阶数，默认为3。
    
    行为:
        - 根据核类型选择适当数据集。
        - 训练SVM模型，预测测试集，输出结果并绘图。
    """
    print(f"Estimating with kernel: {kernel}")
    
    data = Data()
    plotter = Plotting()
    
    if kernel == 'linear':
        X1, y1, X2, y2 = (data.generate_linearly_separable_data() if C == 0 else
                         data.generate_linearly_separable_overlap_data())
    else:
        X1, y1, X2, y2 = data.generate_non_linearly_separable_data()
    
    X_train, y_train, X_test, y_test = data.split_data(X1, y1, X2, y2, test_ratio=0.2)
    
    model = SVM(kernel=kernel, C=C, gamma=gamma, degree=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.correct_predictions = np.sum(y_pred == y_test)
    model.number_of_test_examples = len(X_test)
    model.number_of_train_examples = len(X_train)
    
    formatted_output(model)
    plotter.plot_margin(X_train[y_train == 1], X_train[y_train == -1], model,
                       suffix=f"custom_{kernel}_C_{C}_gamma_{gamma}_degree_{degree}")

def run_sklearn_svm(kernel='linear', C=1.0, gamma='auto', degree=3):
    """
    运行scikit-learn的SVM算法并评估结果。
    
    参数:
        kernel (str): 核函数类型，默认为'linear'。
        C (float): 软间隔参数，默认为1.0。
        gamma (str or float): 高斯核参数，支持'auto'或正浮点数，默认为'auto'。
        degree (int): 多项式核阶数，默认为3。
    
    行为:
        - 根据核类型选择适当数据集。
        - 训练scikit-learn SVM模型，预测测试集并绘图。
    """
    print(f"Estimating with kernel: {kernel}")
    
    data = Data()
    plotter = Plotting()
    
    if kernel == 'linear':
        X1, y1, X2, y2 = (data.generate_linearly_separable_data() if C <= 10 else
                         data.generate_linearly_separable_overlap_data())
    else:
        X1, y1, X2, y2 = data.generate_non_linearly_separable_data()
    
    X_train, y_train, X_test, y_test = data.split_data(X1, y1, X2, y2, test_ratio=0.2)
    
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, coef0=1, gamma=gamma, degree=degree, tol=1e-6)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, gamma=gamma)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    model.fit(X_train, y_train)
    model.correct_predictions = np.sum(model.predict(X_test) == y_test)
    model.number_of_test_examples = len(X_test)
    model.number_of_train_examples = len(X_train)
    model.fit_type = 'sklearn'
    model.kernel = kernel
    
    plotter.plot_margin(X_train[y_train == 1], X_train[y_train == -1], model,
                       suffix=f"sklearn_{kernel}_C_{C}_gamma_{gamma}_degree_{degree}")

if __name__ == "__main__":
    run_custom_svm(kernel='linear')
    run_custom_svm(kernel='linear', C=100)
    run_custom_svm(kernel='polynomial', C=1, degree=3)
    run_custom_svm(kernel='gaussian', gamma=0.5)
    
    run_sklearn_svm(kernel='linear', C=10)
    run_sklearn_svm(kernel='linear', C=100)
    run_sklearn_svm(kernel='poly', C=1, degree=3, gamma=1)
    run_sklearn_svm(kernel='rbf', gamma='auto')
    
    print("Completed")
