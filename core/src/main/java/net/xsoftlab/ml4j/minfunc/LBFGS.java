package net.xsoftlab.ml4j.minfunc;

import java.util.LinkedList;
import java.util.List;

import net.xsoftlab.ml4j.model.BaseModel;

import org.jblas.FloatMatrix;

/**
 * LBFGS
 * 
 * @author 王彦超
 *
 */
public class LBFGS extends MinFunc {

	/**
	 * 初始化
	 * 
	 * @param model
	 *            训练模型
	 */
	public LBFGS(BaseModel model) {
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
	public LBFGS(BaseModel model, int maxIter) {

		this(model);
		this.maxIter = maxIter;
	}

	/**
	 * 最优化theta
	 * 
	 * @return theta
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
		FloatMatrix d, sk, yk, theta1, g1, V, D1;// dk,sk,xk+1,gk

		if (logFlag)
			logger.debug("迭代次数 \t\t步长 \t\t    cost");

		int m = 3;// limit
		int delta, l;
		LinkedList<FloatMatrix> sl = new LinkedList<FloatMatrix>();
		LinkedList<FloatMatrix> yl = new LinkedList<FloatMatrix>();

		d = D0.neg().mmul(g0);// 确定搜索方向
		for (int k = 0; k < maxIter; k++) {

			lamda = Wolfe.lineSearch(model, theta, d);
			sk = d.mul(lamda);
			theta1 = theta.add(sk);

			model.compute(theta1, 3);
			cost1 = model.getCost();
			g1 = model.getGradient();

			if (logFlag) {
				logger.debug("  {} \t   {}   \t {}", new Object[] { k + 1, cost0 - cost1, cost1 });
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
			p = 1f / yk.transpose().mmul(sk).get(0);

			delta = k <= m ? 0 : k - m;
			l = k <= m ? k : m;

			V = I.sub(yk.mmul(sk.transpose()).mul(p));
			D1 = V.transpose().mmul(D0).mmul(V).add(sk.mmul(sk.transpose()).mul(p));

			g0 = g1;
			D0 = D1;
			cost0 = cost1;
			theta = theta1;
		}

		return theta;
	}
}