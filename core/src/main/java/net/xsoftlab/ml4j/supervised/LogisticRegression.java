package net.xsoftlab.ml4j.supervised;

import java.util.List;

import net.xsoftlab.ml4j.util.BGD;
import net.xsoftlab.ml4j.util.MatrixUtil;

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
	 * @param printCost
	 *            是否打印代价函数
	 */
	public LogisticRegression(FloatMatrix x, FloatMatrix y, float alpha, int iterations, boolean printCost) {
		this(x, y, alpha, iterations);

		this.printCost = printCost;
	}

	@Override
	public FloatMatrix function(FloatMatrix theta) {

		return MatrixUtil.sigmoid(x.mmul(theta));
	}

	@Override
	public float computeCost(FloatMatrix theta) {

		FloatMatrix h = function(theta); // sigmoid(X * theta)
		// -y' * log(h)
		FloatMatrix h1 = y.neg().transpose().mmul(MatrixUtil.log(h));
		// (1 - y)' * log(1 - h)
		FloatMatrix h2 = (y.neg().add(1f)).transpose().mmul(MatrixUtil.log(h.neg().add(1f)));

		return 1f / m * (h1.get(0) - h2.get(0));// 1 / m * (h1 - h2)
	}

	@Override
	public FloatMatrix train() {

		logger.info("执行梯度下降...\n");

		BGD bgd = new BGD(this);
		FloatMatrix result = bgd.compute(x, y, theta, alpha, iterations);

		if (printCost) {
			List<FloatMatrix> history = bgd.getHistory();
			for (FloatMatrix theta : history)
				logger.debug("cost history: {}", this.computeCost(theta));
		}

		return result;
	}

}
