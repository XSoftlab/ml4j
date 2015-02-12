package net.xsoftlab.ml4j.supervised;

import java.util.List;

import net.xsoftlab.ml4j.util.GradientDescent;

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
	private float lambda = 0f;// 正则化系数
	private int iterations;// 训练次数

	private int m;// 样本数量
	private boolean printCost = true;// 是否打印代价函数

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
	public LinearRegression(FloatMatrix x, FloatMatrix y, float alpha, int iterations) {
		super();

		this.x = x;
		this.y = y;
		this.alpha = alpha;
		this.iterations = iterations;

		this.m = y.length;
		this.theta = FloatMatrix.rand(x.columns, 1);
	}

	@Override
	public FloatMatrix computeGradient(FloatMatrix theta) {

		FloatMatrix h = x.mmul(theta).sub(y); // x * theta - y
		// x' * h * (alpha / m)
		FloatMatrix h1 = x.transpose().mmul(h);
		FloatMatrix h2 = h1.add(theta.mul(lambda)).mul(alpha / m);

		if (lambda != 0) {
			FloatMatrix h3 = x.getColumn(0).transpose().mmul(h);
			h2.put(0, h3.mul(alpha / m).get(0));
		}

		return h2;
	}

	@Override
	public float computeCost(FloatMatrix theta) {

		FloatMatrix h = x.mmul(theta).sub(y); // x * theta - y
		FloatMatrix h1 = h.transpose().mmul(h);// h' * h

		return 1f / (2 * m) * h1.get(0);// 1/2m * h1
	}

	@Override
	public FloatMatrix train() {

		logger.info("执行梯度下降...\n");

		GradientDescent gd = new GradientDescent(this, printCost);
		FloatMatrix result = gd.compute(theta, iterations);

		if (printCost) {
			List<FloatMatrix> history = gd.getHistory();
			for (FloatMatrix theta : history)
				logger.debug("cost history: {}", this.computeCost(theta));
		}

		return result;
	}

	public void setPrintCost(boolean printCost) {
		this.printCost = printCost;
	}

}
