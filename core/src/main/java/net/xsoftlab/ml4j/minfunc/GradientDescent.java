package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.BaseModel;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 梯度下降
 * 
 * @author 王彦超
 *
 */
public class GradientDescent {

	private BaseModel model;// 训练模型
	private float alpha = 0.01f;// 训练速度
	private int maxIter = 500;// 最大训练次数
	private float epsilon = (float) 1e-6;// 精度阀值

	private FloatMatrix theta;// 参数
	private boolean logFlag = true;// 是否打印过程日志

	Logger logger = LoggerFactory.getLogger(this.getClass());

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

	/**
	 * 计算theta
	 * 
	 * @return theta
	 */
	public FloatMatrix compute() {

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

	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;
	}

	public void setTheta(FloatMatrix theta) {
		this.theta = theta;
	}

	public void setCostFlag(boolean costFlag) {
		this.logFlag = costFlag;
	}

}
