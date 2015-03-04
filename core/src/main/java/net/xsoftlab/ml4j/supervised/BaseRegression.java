package net.xsoftlab.ml4j.supervised;

import org.jblas.FloatMatrix;

/**
 * 回归算法基类
 * 
 * @author 王彦超
 *
 */
public interface BaseRegression {

	/**
	 * hypothesis(假设)函数
	 * 
	 * @param theta
	 *            参数
	 * @return 计算好的矩阵
	 */
	FloatMatrix function(FloatMatrix theta);

	/**
	 * 计算梯度
	 * 
	 * @param theta
	 *            参数
	 * @return theta
	 */
	FloatMatrix computeGradient(FloatMatrix theta);

	/**
	 * 计算代价(目标)函数
	 * 
	 * @param theta
	 *            参数
	 * @return 代价值
	 */
	float computeCost(FloatMatrix theta);

	/**
	 * 训练
	 * 
	 * @return 训练结果
	 */
	FloatMatrix train();

}
