package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.util.MatrixUtil;

import org.jblas.FloatMatrix;

/**
 * 逻辑回归模型
 * 
 * @author 王彦超
 * 
 */
public class LogisticRegression extends BaseModel {

	/**
	 * 初始化
	 * 
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 */
	public LogisticRegression(FloatMatrix x, FloatMatrix y) {
		super(x, y);
	}

	/**
	 * 初始化
	 * 
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 * @param lambda
	 *            正则系数
	 */
	public LogisticRegression(FloatMatrix x, FloatMatrix y, float lambda) {
		super(x, y, lambda);
	}

	@Override
	public FloatMatrix function(FloatMatrix theta) {

		return MatrixUtil.sigmoid(x.mmul(theta));
	}

	@Override
	public FloatMatrix gradient(FloatMatrix theta) {

		// sigmoid(X * theta) - y
		FloatMatrix h = function(theta).sub(y);
		// x' * h * (alpha / m)
		FloatMatrix h1 = x.transpose().mmul(h);
		FloatMatrix h2 = h1.add(theta.mul(lambda));

		if (lambda != 0) {
			FloatMatrix h3 = x.getColumn(0).transpose().mmul(h);
			h2.put(0, h3.get(0));
		}

		return h2.div(m);
	}

	@Override
	public float cost(FloatMatrix theta) {

		// sigmoid(X * theta)
		FloatMatrix h = function(theta);
		// -y' * log(h)
		FloatMatrix h1 = y.neg().transpose().mmul(MatrixUtil.log(h));
		// (1 - y)' * log(1 - h)
		FloatMatrix h2 = (y.neg().add(1f)).transpose().mmul(MatrixUtil.log(h.neg().add(1f)));
		float cost = 1f / m * (h1.get(0) - h2.get(0));// 1 / m * (h1 - h2)

		FloatMatrix theta1 = theta.getRange(1, theta.length);
		if (lambda != 0) {
			float cost1 = lambda / (2 * m) * theta1.transpose().mmul(theta1).get(0);
			cost += cost1;
		}

		return cost;
	}

}
