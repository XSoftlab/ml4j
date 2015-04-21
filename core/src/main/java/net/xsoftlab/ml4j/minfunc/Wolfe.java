package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.supervised.BaseModel;

import org.jblas.FloatMatrix;

/**
 * Wolfe 模糊一维搜索
 * 
 * @author 王彦超
 *
 */
public class Wolfe {

	public static float lineSearch(BaseModel model, FloatMatrix theta, FloatMatrix d) {

		// 默认设置
		float mu = 0.1f, sigma = 0.5f;
		float a = 0, b = Float.MAX_VALUE, alpha = 1;

		model.compute(theta, 3);
		float cost0 = model.getCost();// 初始cost
		FloatMatrix g0 = model.getGradient();// 初始梯度

		FloatMatrix theta1 = theta.add(d.mul(alpha));// 求解最优步长因子 alpha
		model.compute(theta1, 3);
		float cost1 = model.getCost();
		FloatMatrix g1 = model.getGradient();

		while (cost0 - cost1 < g0.transpose().mmul(d).mul(alpha).mul(-mu).get(0)) {
			b = alpha;
			alpha = 0.5f * (alpha + a);
			theta1 = theta.add(d.mul(alpha));
			model.compute(theta1, 1);
			cost1 = model.getCost();
		}

		while (g1.transpose().mmul(d).get(0) < g0.transpose().mmul(d).mul(sigma).get(0)) {
			a = alpha;
			alpha = 2f * alpha < 0.5f * (a + b) ? 2f * alpha : 0.5f * (a + b);
			theta1 = theta.add(d.mul(alpha));
			model.compute(theta1, 2);
			g1 = model.getGradient();
		}

		return alpha;
	}
}
