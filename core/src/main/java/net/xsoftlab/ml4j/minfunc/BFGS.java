package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.BaseModel;

import org.jblas.FloatMatrix;

/**
 * BFGS
 * 
 * @author 王彦超
 *
 */
public class BFGS extends MinFunc {

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
	 * 最优化theta
	 * 
	 * @return theta
	 * @see http://blog.csdn.net/itplus/article/details/21897443
	 */
	@Override
	public FloatMatrix compute() {

		int n = theta.length; // theat长度
		FloatMatrix I = FloatMatrix.eye(n);// 单位矩阵
		FloatMatrix D0 = I;// 初始化D0

		model.compute(theta, 3);
		float cost0 = model.getCost();// 初始cost
		FloatMatrix g0 = model.getGradient();// 初始梯度

		float cost1, p, lamda;// lamda:一维搜索步长
		FloatMatrix d, s, g1, yk, V, D1;// dk,sk,xk+1,gk

		if (logFlag)
			logger.debug("迭代次数 \t\t步长 \t\t    cost");

		for (int k = 0; k < maxIter; k++) {

			d = D0.neg().mmul(g0);// 确定搜索方向
			lamda = Wolfe.lineSearch(model, theta, d);
			s = d.mul(lamda);
			theta = theta.add(s);

			model.compute(theta, 3);
			cost1 = model.getCost();
			g1 = model.getGradient();

			if (logFlag) {
				logger.debug("  {} \t   {}   \t {}", new Object[] { k + 1, cost0 - cost1, cost1 });
			}

			if (g1.transpose().mmul(g1).get(0) < epsilon) {
				logger.info("\n已达到梯度精度阀值.\n");
				break;
			}

			if (cost0 - cost1 < epsilon) {
				logger.info("\n已达到cost精度阀值.\n");
				break;
			}

			yk = g1.sub(g0);
			p = 1f / yk.transpose().mmul(s).get(0);
			V = I.sub(yk.mmul(s.transpose()).mul(p));
			D1 = V.transpose().mmul(D0).mmul(V).add(s.mmul(s.transpose()).mul(p));

			g0 = g1;
			D0 = D1;
			cost0 = cost1;
		}

		return theta;
	}
}