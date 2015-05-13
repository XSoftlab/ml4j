# ml4j - Machine learning for java

### 机器学习算法
* 监督学习
	* 线性回归 - LinearRegression
	* 逻辑回归 - LogisticRegression
	* 神经网络 - NeuralNetworks
	* 支撑向量机 - SVM(libSVM)
	* 协同过滤 - CollaborativeFiltering
* 无监督学习
	* K-means
	* PCA
	* 异常检测 - AnomalyDetection

### 最优化算法
* 梯度下降
* BFGS
* LBFGS

### 测试实例
* 监督学习
	* 线性回归 - 房价预测
	* 逻辑回归 - 手写数字mnist识别（0,1）
	* OneVsAll - 手写数字mnist识别
	* OneVsAll 多线程版 - 挑战CPU的极限
	* 神经网络 - 手写数字mnist识别
	* SVM - 垃圾邮件分类,4000训练，1000测试，测试集正确率98.9%
* 无监督学习
	* K-mean - 将24位的有上千种颜色的RGB图象压缩至16种颜色
	* PCA - 人脸数据降维
	* AnomalyDetection 异常检测
* 应用
	* 推荐系统 - 基于协同过滤的电影推荐系统