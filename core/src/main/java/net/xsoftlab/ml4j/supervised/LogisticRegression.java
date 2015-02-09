package net.xsoftlab.ml4j.supervised;

import java.util.List;

import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.SGD;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 逻辑回归
 * 
 * @author 王彦超
 * 
 */
public class LogisticRegression implements BaseRegression {

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
	public LogisticRegression(FloatMatrix x, FloatMatrix y, float alpha, int iterations) {
		super();

		this.x = x;
		this.y = y;
		this.alpha = alpha;
		this.iterations = iterations;

		this.m = y.length;
		this.theta = FloatMatrix.rand(x.columns, 1);
	}

	/**
	 * 初始化
	 * 
	 * @param lambda
	 *            正则系数
	 */
	public LogisticRegression(FloatMatrix x, FloatMatrix y, float alpha, int iterations, float lambda) {
		this(x, y, alpha, iterations);

		this.lambda = lambda;
	}

	@Override
	public FloatMatrix function(FloatMatrix theta) {

		return MatrixUtil.sigmoid(x.mmul(theta));
	}

	@Override
	public FloatMatrix function(FloatMatrix theta, int i) {

		return MatrixUtil.sigmoid(x.getRow(i).mmul(theta));
	}

	@Override
	public float computeCost(FloatMatrix theta) {

		FloatMatrix h = function(theta); // sigmoid(X * theta)
		// -y' * log(h)
		FloatMatrix h1 = y.neg().transpose().mmul(MatrixUtil.log(h));
		// (1 - y)' * log(1 - h)
		FloatMatrix h2 = (y.neg().add(1f)).transpose().mmul(MatrixUtil.log(h.neg().add(1f)));

		FloatMatrix theta1 = theta.getRange(1, theta.length);
		float cost = 1f / m * (h1.get(0) - h2.get(0));// 1 / m * (h1 - h2)
		if (lambda != 0)
			cost += lambda / (2 * m) * theta1.transpose().mmul(theta1).get(0);

		return cost;
	}

	@Override
	public FloatMatrix train() {

		logger.info("执行梯度下降...\n");

		SGD sgd = new SGD(this, printCost);
		FloatMatrix result = sgd.computeWithLambda(x, y, theta, lambda, alpha, iterations);

		if (printCost) {
			List<FloatMatrix> history = sgd.getHistory();
			for (FloatMatrix theta : history)
				logger.debug("cost history: {}", this.computeCost(theta));
		}

		return result;
	}

	public void setLambda(float lambda) {
		this.lambda = lambda;
	}

	public void setPrintCost(boolean printCost) {
		this.printCost = printCost;
	}

}
