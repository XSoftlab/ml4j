package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.util.MathUtil;

import org.jblas.FloatMatrix;

/**
 * 线性回归模型
 * 
 * @author 王彦超
 * 
 */
public class LinearRegression extends BaseModel {

	/**
	 * 初始化
	 * 
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 */
	public LinearRegression(FloatMatrix x, FloatMatrix y) {
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
	public LinearRegression(FloatMatrix x, FloatMatrix y, float lambda) {
		this(x, y);

		this.lambda = lambda;
	}

	@Override
	public void compute(FloatMatrix theta, int flag) {

		FloatMatrix h = x.mmul(theta).sub(y); // x * theta - y

		if (flag == 1 || flag == 3) {
			FloatMatrix h1 = h.transpose().mmul(h);// h' * h
			this.cost = h1.get(0);

			if (lambda != 0) {
				FloatMatrix theta1 = theta.getRange(1, theta.length);
				float cost1 = lambda * theta1.transpose().mmul(theta1).get(0);
				cost += cost1;
			}
			cost = 1f / (2 * m) * cost;
		}

		if (flag == 2 || flag == 3) {
			FloatMatrix h1 = x.transpose().mmul(h);// x' * h * (alpha / m)
			FloatMatrix h2 = h1.add(theta.mul(lambda));

			if (lambda != 0) {
				FloatMatrix h3 = x.getColumn(0).transpose().mmul(h);
				h2.put(0, h3.get(0));
			}
			this.gradient = h2.div(m);
		}
	}

	@Override
	public void checkGradients() {
		// TODO Auto-generated method stub

	}

	@Override
	public float evaluate(FloatMatrix theta) {

		return evaluate(theta, x, y);
	}

	@Override
	public float evaluate(FloatMatrix theta, FloatMatrix x, FloatMatrix y) {

		return MathUtil.std(x.mmul(theta), y);
	}

	@Override
	public FloatMatrix getInitTheta() {

		return FloatMatrix.rand(x.columns, 1).mul(0.001f);
	}

}
