package net.xsoftlab.ml4j.model;

import org.jblas.FloatMatrix;

/**
 * 模型接口
 * 
 * @author 王彦超
 *
 */
public abstract class BaseModel {

	protected FloatMatrix x;// 特征值
	protected FloatMatrix y;// 标签
	protected float lambda = 0f;// 正则化系数

	protected int m;// 样本数量
	protected float cost;// cost
	protected FloatMatrix gradient;// 梯度

	/**
	 * 计算梯度/代价(目标)函数
	 * 
	 * @param theta
	 *            参数
	 * @param flag
	 *            1.计算梯度 2.计算cost 3.计算全部
	 * @return 梯度/cost
	 */
	public abstract void compute(FloatMatrix theta, int flag);

	/**
	 * 获取初始theta值
	 * 
	 * @return theta
	 */
	public abstract FloatMatrix getInitTheta();

	public float getCost() {
		return cost;
	}

	public FloatMatrix getGradient() {
		return gradient;
	}
}
