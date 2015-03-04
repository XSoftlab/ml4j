package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.model.BaseModel;

import org.jblas.FloatMatrix;

/**
 * 线性回归
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

	@Override
	public FloatMatrix function(FloatMatrix theta) {

		return x.mmul(theta);
	}

	@Override
	public FloatMatrix gradient(FloatMatrix theta) {

		FloatMatrix h = function(theta).sub(y); // x * theta - y
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

		FloatMatrix h = function(theta).sub(y); // x * theta - y
		FloatMatrix h1 = h.transpose().mmul(h);// h' * h
		float cost = h1.get(0);

		FloatMatrix theta1 = theta.getRange(1, theta.length);
		if (lambda != 0) {
			float cost1 = lambda * theta1.transpose().mmul(theta1).get(0);
			cost += cost1;
		}

		return 1f / (2 * m) * cost;
	}

}
