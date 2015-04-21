package net.xsoftlab.ml4j.common;

import net.xsoftlab.ml4j.model.supervised.BaseModel;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 梯度校验
 * 
 * @author 王彦超
 *
 */
public class GradientChecking {

	protected BaseModel model;// 训练模型
	protected int maxIter = 100;// 最大校验次数
	private float delta = 1e-3f;// 梯度校验区间

	protected FloatMatrix theta;// 参数

	Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 */
	public GradientChecking(BaseModel model) {
		super();
		this.model = model;

		this.theta = model.getInitTheta();
	}

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 * @param maxIter
	 *            最大训练次数
	 */
	public GradientChecking(BaseModel model, int maxIter) {

		this(model);
		this.maxIter = maxIter;
	}

	public void check() {

		float gest, error, avgError, sumError = 0;
		float cost1, cost2;

		model.compute(theta, 2);
		FloatMatrix gradient = model.getGradient();// 初始梯度

		FloatMatrix t0 = theta.dup();
		FloatMatrix t1 = theta.dup();

		maxIter = maxIter < theta.length ? maxIter : theta.length;

		logger.debug("迭代次数 \t\t误差 \t\tgradient \tgest");
		for (int i = 0; i < maxIter; i++) {
			t0.put(i, t0.get(i) - delta);
			t1.put(i, t1.get(i) + delta);

			model.compute(t0, 1);
			cost1 = model.getCost();

			model.compute(t1, 1);
			cost2 = model.getCost();

			gest = (cost2 - cost1) / (2 * delta);
			error = Math.abs(gradient.get(i) - gest);

			sumError += error;
			t0.put(i, t0.get(i) + delta);
			t1.put(i, t1.get(i) - delta);

			logger.debug("  {} \t   {}\t   {}\t   {}", new Object[] { i + 1, error, gradient.get(i), gest });
		}

		avgError = sumError / (float) maxIter;

		logger.debug("平均误差： {}", avgError);
	}
}
