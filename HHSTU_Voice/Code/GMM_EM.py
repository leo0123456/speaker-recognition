
import copy

import numpy as np

from scipy.stats import multivariate_normal

class Gmmly:
    def __init__(self):
        pass

    ############################################################
    # 数据预处理
    # X为样本矩阵
    # 将数据进行极差归一化处理
    ############################################################

    def scale_data(self,x):
        for i in range(x.shape[1]):
            max_ = x[:, i].max()
            min_ = x[:, i].min()
            x[:, i] = (x[:, i] - min_) / (max_ - min_)
        return x

    ############################################################
    # 初始化模型参数
    # shape为样本矩阵x的维数（样本数，特征数）
    # k为模型的个数
    # mu, cov, alpha分别为模型的均值、协方差以及混合系数
    ############################################################

    def init_params(self,shape, k):
        n,d= shape
        self.K=k
        self.mu = np.random.rand(k, d)
        self.cov = np.array([np.eye(d)] * k) * 0.1
        self.alpha = np.array([1.0 / k] * k)
        return self.mu, self.cov, self.alpha

    ############################################################
    # 第i个模型的高斯密度分布函数
    # x 为样本矩阵，行数等于样本数，列数等于特征数
    # mu_i, cov_i分别为第i个模型的均值、协方差参数
    # 返回样本在该模型下的概率密度值
    ############################################################

    def phi(self,Y, mu_k, cov_k):
        norm = multivariate_normal(mean=mu_k, cov=cov_k)
        return norm.pdf(Y)




    ############################################################
    # E步：计算每个模型对样本的响应度
    # x 为样本矩阵，行数等于样本数，列数等于特征数
    # mu为均值矩阵， cov为协方差矩阵
    # alpha为各模型混合系数组成的一维矩阵
    ############################################################

    def expectation(self,x,):
        # 样本数，模型数
        n, k = x.shape[0], self.alpha.shape[0]
        a=self.mu
        b=self.cov
        # 计算各模型下所有样本出现的概率矩阵prob，行对应第i个样本，列对应第K个模型
        prob = []
        for j in range(n):
            prob_j = []    # 依次求每个样本对K个分模型的响应度
            for k in range(self.K):
                aa=self.alpha[k] * self.phi(x[j], self.mu[k], self.cov[k])
                prob_j.append(self.alpha[k] * self.phi(x[j], self.mu[k], self.cov[k]))
            s = sum(prob_j)+1e-3
            prob_j = [item/s for item in prob_j]
            prob.append(prob_j)
        prob = np.array(prob)
        return prob

    ############################################################
    # M步：迭代模型参数
    ############################################################

    def maximization(self,x, prob):
        # 样本数，特征数
        n, d = x.shape
        # 模型数
        old_alpha = copy.copy(self.alpha)
        k = prob.shape[1]

        # 初始化模型参数
        mu = np.zeros((k, d))
        cov = []
        alpha = np.zeros(k)

        # 更新每个模型的参数
        for i in range(k):
            gamma_k = prob[:, i]
            SUM = np.sum(gamma_k)+1e-4
            # 更新权重
            self.alpha[i] = SUM / n  # 更新权重
            # 更新均值向量
            new_mu = sum([gamma * y for gamma, y in zip(gamma_k, x)]) / SUM  # 1*d
            self.mu[i] = new_mu
            # 更新协方差阵
            delta_ = x - new_mu   # n*d
            self.cov[i] = sum([gamma * (np.outer(np.transpose([delta]), delta)) for gamma, delta in zip(gamma_k, delta_)]) / SUM  # d*d
        alpha_delta = self.alpha - old_alpha
        gamma_all_final = prob
        return self.mu, self.cov, self.alpha, gamma_all_final

    ############################################################
    # 高斯混合模型EM算法
    # x为样本矩阵，k为模型个数，times为模型迭代次数
    ############################################################

    def gmm_em(self,dataset,k, times):
        # 数据归一化处理
        x = self.scale_data(dataset)
        # 初始化模型参数
        mu, cov, alpha = self.init_params(x.shape, k)

        # 迭代模型参数
        for i in range(times):
            print("这是第{0}次",i)
            prob = self.expectation(x)
            mu, cov, alpha, gamma = self.maximization(x, prob)



        return gamma



