package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;

import org.jblas.FloatMatrix;

/**
 * 模型接口
 * 
 * @author 王彦超
 *
 */
public abstract class BaseModel {

	protected FloatMatrix x;// 特征矩阵
	protected FloatMatrix y;// 标签
	protected float lambda = 0f;// 正则化系数

	protected int m;// 样本数量
	protected float cost;// cost
	protected FloatMatrix gradient;// 梯度

	protected boolean checkFlag = false;// 梯度校验开关

	/**
	 * 计算梯度/代价(目标)函数
	 * 
	 * @param theta
	 *            参数
	 * @param flag
	 *            1.计算cost 2.计算梯度 3.计算全部
	 * @return 梯度/cost
	 */
	public abstract Object compute(FloatMatrix theta, int flag);

	/**
	 * 计算最优化theta - 使用默认的minFunc(l-bfgs)
	 * 
	 * @return theta
	 */
	public FloatMatrix train() {
		return train(new LBFGS(this));
	}

	/**
	 * 计算最优化theta - 使用默认的minFunc(l-bfgs)
	 * 
	 * @param maxIter 最大迭代次数
	 * 
	 * @return theta
	 */
	public FloatMatrix train(int maxIter) {
		return train(new LBFGS(this, maxIter));
	}

	/**
	 * 计算最优化theta
	 * 
	 * @return theta
	 */
	public FloatMatrix train(MinFunc minFunc) {
		return minFunc.train();
	}

	/**
	 * 准确度评测
	 * 
	 * @param theta
	 *            训练好的theta
	 * @return 准确度
	 */
	public float evaluate(FloatMatrix theta) {

		return evaluate(theta, x, y);
	}

	/**
	 * 准确度评测
	 * 
	 * @param theta
	 *            训练好的theta
	 * @param x
	 *            测试集
	 * @param y
	 *            标签
	 * @return 准确度
	 */
	public abstract float evaluate(FloatMatrix theta, FloatMatrix x, FloatMatrix y);

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
