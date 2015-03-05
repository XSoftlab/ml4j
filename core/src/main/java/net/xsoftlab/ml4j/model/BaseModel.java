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

	public BaseModel(FloatMatrix x, FloatMatrix y) {
		super();
		this.x = x;
		this.y = y;

		this.m = y.length;
	}

	public BaseModel(FloatMatrix x, FloatMatrix y, float lambda) {
		this(x, y);

		this.lambda = lambda;
	}

	/**
	 * hypothesis(假设)函数
	 * 
	 * @param theta
	 *            参数
	 * @return 计算好的矩阵
	 */
	public abstract FloatMatrix function(FloatMatrix theta);

	/**
	 * 计算梯度
	 * 
	 * @param theta
	 *            参数
	 * @return theta
	 */
	public abstract FloatMatrix gradient(FloatMatrix theta);

	/**
	 * 计算代价(目标)函数
	 * 
	 * @param theta
	 *            参数
	 * @return 代价值
	 */
	public abstract float cost(FloatMatrix theta);

	/**
	 * 获取初始theta值
	 * 
	 * @return theta
	 */
	public FloatMatrix theta() {

		return FloatMatrix.zeros(x.columns, 1);
	}
}
