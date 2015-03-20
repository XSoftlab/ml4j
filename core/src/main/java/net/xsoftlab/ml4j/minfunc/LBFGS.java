package net.xsoftlab.ml4j.minfunc;

import java.util.LinkedList;

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
		FloatMatrix D0 = FloatMatrix.eye(n);// 单位矩阵

		model.compute(theta, 3);
		float cost0 = model.getCost();// 初始cost
		FloatMatrix gk = model.getGradient();// 初始梯度

		float cost1, lamda;// lamda:一维搜索步长
		FloatMatrix d, sk, yk, gk1;// dk,sk,xk+1,gk

		if (logFlag)
			logger.debug("迭代次数 \t\t步长 \t\t    cost");

		int l, m = 3;// limit
		float alpha, beta;
		FloatMatrix ql;
		LinkedList<FloatMatrix> sl = new LinkedList<FloatMatrix>();
		LinkedList<FloatMatrix> yl = new LinkedList<FloatMatrix>();
		LinkedList<Float> pl = new LinkedList<Float>();
		Float[] alphal = new Float[m];

		for (int k = 0; k < maxIter; k++) {

			l = k <= m ? k : m;
			ql = gk;

			for (int i = l - 1; i >= 0; i--) {
				alpha = sl.get(i).transpose().mmul(ql).mul(pl.get(i)).get(0);
				ql = ql.sub(yl.get(i).mul(alpha));
				alphal[i] = alpha;
			}

			d = D0.mmul(ql);
			for (int i = 0; i < l; i++) {
				beta = yl.get(i).transpose().mmul(d).mul(pl.get(i)).get(0);
				d = sl.get(i).mul(alphal[i] - beta).add(d);
			}

			d = d.neg();
			lamda = Wolfe.lineSearch(model, theta, d);
			sk = d.mul(lamda);
			theta = theta.add(sk);

			model.compute(theta, 3);
			cost1 = model.getCost();
			gk1 = model.getGradient();

			if (logFlag) {
				logger.debug("  {} \t   {}   \t {}", new Object[] { k + 1, cost0 - cost1, cost1 });
			}

			if (gk1.transpose().mmul(gk1).get(0) < epsilon) {
				logger.info("\n已达到梯度精度阀值.\n");
				break;
			}

			if (cost0 - cost1 < epsilon) {
				logger.info("\n已达到cost精度阀值.\n");
				break;
			}

			yk = gk1.sub(gk);

			yl.add(yk);
			if (yl.size() > m)
				yl.removeFirst();

			sl.add(sk);
			if (sl.size() > m)
				sl.removeFirst();

			pl.add(1f / yk.transpose().mmul(sk).get(0));
			if (pl.size() > m)
				pl.removeFirst();

			gk = gk1;
			cost0 = cost1;
		}

		return theta;
	}
}