package net.xsoftlab.ml4j.regression;

import java.util.List;

import net.xsoftlab.ml4j.common.BGD;
import net.xsoftlab.ml4j.common.Ml4jException;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 线性回归
 * 
 * @author 王彦超
 * 
 */
public class LinearRegression implements BaseRegression {

	private FloatMatrix x;// 特征值
	private FloatMatrix y;// 标签
	private FloatMatrix theta;// 参数
	private float alpha = 0.01f;// 训练速度
	private int iterations;// 训练次数

	private int m;// 样本数量

	Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * 初始化
	 * 
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 * @param theta
	 *            参数
	 * @param alpha
	 *            训练速度
	 * @param iterations
	 *            训练次数
	 */
	public LinearRegression(FloatMatrix x, FloatMatrix y, FloatMatrix theta, float alpha, int iterations) {
		super();

		if (!y.isVector()) {
			logger.error("y 必须是向量.", new Ml4jException("y 必须是向量."));
		}

		this.x = x;
		this.y = y;
		this.theta = theta;
		this.alpha = alpha;
		this.iterations = iterations;

		this.m = y.length;
	}

	@Override
	public float computeCost(FloatMatrix theta) {

		FloatMatrix h = x.mmul(theta).sub(y); // x * theta - y
		FloatMatrix h1 = h.transpose().mmul(h);// h' * h

		return 1 / (2 * m) * h1.get(0);// 1/2m * h1
	}

	@Override
	public FloatMatrix train() {

		logger.debug("执行梯度下降...");

		BGD bgd = new BGD();
		List<FloatMatrix> history = bgd.getHistory();
		for (FloatMatrix theta : history)
			logger.debug("theta history : {}", computeCost(theta));

		return bgd.compute(x, y, theta, alpha, iterations);
	}

}
