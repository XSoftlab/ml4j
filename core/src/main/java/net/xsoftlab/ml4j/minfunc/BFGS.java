package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.BaseModel;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * BFGS
 * 
 * @author 王彦超
 *
 */
public class BFGS {

	private BaseModel model;// 训练模型
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
	 */
	public BFGS(BaseModel model) {
		super();
		this.model = model;

		this.theta = model.getInitTheta();
	}

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 * @param maxIter
	 *            最大训练次数
	 */
	public BFGS(BaseModel model, int maxIter) {

		this(model);
		this.maxIter = maxIter;
	}

	/**
	 * 计算theta
	 * 
	 * @return theta
	 * @see http://blog.csdn.net/itplus/article/details/21897443
	 */
	public FloatMatrix compute() {

		int n = theta.length; // theat长度
		FloatMatrix I = FloatMatrix.eye(n);// 单位矩阵
		FloatMatrix D0 = I;// 初始化D0

		model.compute(theta, 3);
		float cost0 = model.getCost();// 初始cost
		FloatMatrix g0 = model.getGradient();// 初始梯度

		float cost1, p, lamda;// lamda:一维搜索步长
		FloatMatrix d, s, theta1, g1, yk, V, D1;// dk,sk,xk+1,gk

		if (logFlag)
			logger.debug("迭代次数 \t\t步长 \t\t    cost");

		for (int i = 0; i < maxIter; i++) {

			d = D0.neg().mmul(g0);// 确定搜索方向
			lamda = Wolfe.lineSearch(model, theta, d);
			s = d.mul(lamda);
			theta1 = theta.add(s);

			model.compute(theta1, 3);
			cost1 = model.getCost();// 初始cost
			g1 = model.getGradient();// 初始梯度

			if (logFlag) {
				logger.debug("  {} \t   {}   \t {}", new Object[] { i + 1, cost0 - cost1, cost1 });
			}

			if (g1.transpose().mmul(g1).get(0) < epsilon) {
				logger.info("\n已达到梯度精度阀值.\n");
				return theta1;
			}

			if (cost0 - cost1 < epsilon) {
				logger.info("\n已达到cost精度阀值.\n");
				return theta1;
			}

			yk = g1.sub(g0);
			p = 1f / yk.transpose().mmul(s).get(0);
			V = I.sub(yk.mmul(s.transpose()).mul(p));
			D1 = V.transpose().mmul(D0).mmul(V).add(s.mmul(s.transpose()).mul(p));

			g0 = g1;
			D0 = D1;
			cost0 = cost1;
			theta = theta1;
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

class Wolfe {

	public static float lineSearch(BaseModel model, FloatMatrix theta, FloatMatrix d) {

		// 默认设置
		float mu = 0.1f, sigma = 0.5f;
		float a = 0, b = Float.MAX_VALUE, alpha = 1;

		float cost0 = model.cost(theta);// 初始cost
		FloatMatrix g0 = model.gradient(theta);// 初始梯度
		FloatMatrix theta1 = theta.add(d.mul(alpha));// 求解最优步长因子 alpha
		float cost1 = model.cost(theta1);
		FloatMatrix g1 = model.gradient(theta1);

		while (cost0 - cost1 < g0.transpose().mmul(d).mul(alpha).mul(-mu).get(0)) {
			b = alpha;
			alpha = 0.5f * (alpha + a);
			theta1 = theta.add(d.mul(alpha));
			cost1 = model.cost(theta1);
		}

		while (g1.transpose().mmul(d).get(0) < g0.transpose().mmul(d).mul(sigma).get(0)) {
			a = alpha;
			alpha = 2f * alpha < 0.5f * (a + b) ? 2f * alpha : 0.5f * (a + b);
			theta1 = theta.add(d.mul(alpha));
			g1 = model.gradient(theta1);
		}

		return alpha;
	}
}
