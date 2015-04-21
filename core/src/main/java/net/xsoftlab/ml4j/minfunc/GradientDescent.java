package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.supervised.BaseModel;

import org.jblas.FloatMatrix;

/**
 * 梯度下降
 * 
 * @author 王彦超
 *
 */
public class GradientDescent extends MinFunc {

	private float alpha = 0.01f;// 训练速度

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 * @param alpha
	 *            训练速度
	 */
	public GradientDescent(BaseModel model, float alpha) {
		super();
		this.model = model;
		this.alpha = alpha;

		this.theta = model.getInitTheta();
	}

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 * @param alpha
	 *            训练速度
	 * @param maxIter
	 *            最大训练次数
	 */
	public GradientDescent(BaseModel model, float alpha, int maxIter) {

		this(model, alpha);
		this.maxIter = maxIter;
	}

	@Override
	public FloatMatrix train() {

		float step, cost0 = 0, cost1 = 0;

		if (logFlag)
			logger.debug("迭代次数 \t\t步长 \t\t    cost");

		for (int i = 0; i < maxIter; i++) {

			model.compute(theta, 3);
			theta = theta.sub(model.getGradient().mul(alpha));
			cost1 = model.getCost();
			step = Math.abs(cost0 - cost1);

			if (logFlag) {
				logger.debug("  {} \t   {}   \t {}", new Object[] { i + 1, step, cost1 });
			}
			if (step < epsilon) {
				logger.info("\n已达到精度阀值.\n");
				return theta;
			}

			cost0 = cost1;
		}

		return theta;
	}
}
