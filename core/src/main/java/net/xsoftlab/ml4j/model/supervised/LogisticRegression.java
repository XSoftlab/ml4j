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
		super();
		this.x = x;
		this.y = y;

		this.m = y.length;
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
		this(x, y);

		this.lambda = lambda;
	}

	@Override
	public void compute(FloatMatrix theta, int flag) {

		// sigmoid(X * theta)
		FloatMatrix h = MatrixUtil.sigmoid(x.mmul(theta));

		if (flag == 1 || flag == 3) {
			// -y' * log(h)
			FloatMatrix h1 = y.neg().transpose().mmul(MatrixUtil.log(h));
			// (1 - y)' * log(1 - h)
			FloatMatrix h2 = (y.neg().add(1f)).transpose().mmul(MatrixUtil.log(h.neg().add(1f)));
			this.cost = 1f / m * (h1.get(0) - h2.get(0));// 1 / m * (h1 - h2)

			if (lambda != 0) {
				FloatMatrix theta1 = theta.getRange(1, theta.length);
				float cost1 = lambda / (2 * m) * theta1.transpose().mmul(theta1).get(0);
				cost += cost1;
			}
		}

		if (flag == 2 || flag == 3) {
			h = h.sub(y);// sigmoid(X * theta) - y
			// x' * h * (alpha / m)
			FloatMatrix h1 = x.transpose().mmul(h);
			FloatMatrix h2 = h1.add(theta.mul(lambda));

			if (lambda != 0) {
				FloatMatrix h3 = x.getColumn(0).transpose().mmul(h);
				h2.put(0, h3.get(0));
			}

			this.gradient = h2.div(m);
		}
	}

	@Override
	public float evaluate(FloatMatrix theta) {
		
		return evaluate(theta, x, y);
	}

	@Override
	public float evaluate(FloatMatrix theta, FloatMatrix x, FloatMatrix y) {
		
		FloatMatrix y1 = MatrixUtil.sigmoid(x.mmul(theta));
		return y1.ge(0.5f).eq(y).mean() * 100;
	}

	@Override
	public FloatMatrix getInitTheta() {

		return FloatMatrix.rand(x.columns, 1).mul(0.001f);
	}
}
