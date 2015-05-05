package net.xsoftlab.ml4j.model.unsupervised;

import net.xsoftlab.ml4j.util.MatrixUtil;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Solve;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 异常检测
 * 
 * @author X
 *
 * @data 2015年5月6日
 */
public class AnomalyDetection {

	private FloatMatrix mu;// 高斯参数 mu
	private FloatMatrix sigma;// 高斯参数 sigma

	private FloatMatrix x;// 特征矩阵
	private FloatMatrix xVal;// 特征矩阵
	private FloatMatrix yVal;// 特征矩阵

	private float epsilon;// 异常阈值
	private float f1;// f1 score

	private Logger logger = LoggerFactory.getLogger(AnomalyDetection.class);

	/**
	 * 初始化异常检测
	 * 
	 * @param x
	 * @param xVal
	 * @param yVal
	 */
	public AnomalyDetection(FloatMatrix x, FloatMatrix xVal, FloatMatrix yVal) {
		super();
		this.x = x;
		this.xVal = xVal;
		this.yVal = yVal;

		// 初始化过程中直接计算好高斯参数
		estimateGaussian();
	}

	/**
	 * 计算高斯参数
	 * 
	 * @return mu
	 */
	public void estimateGaussian() {

		mu = x.columnMeans();
		sigma = MatrixUtil.var(x, false, 1);
	}

	/**
	 * 计算多元高斯分布概率
	 * 
	 * @return 多元高斯分布概率
	 */
	public FloatMatrix multivariateGaussian(FloatMatrix matrix) {

		int k = mu.length;
		FloatMatrix sigma2 = sigma.dup();
		if (sigma2.rows == 1 || sigma2.columns == 1)
			sigma2 = FloatMatrix.diag(sigma2);

		FloatMatrix x1 = matrix.subRowVector(mu);
		float v1 = (float) MatrixFunctions.pow(2 * Math.PI, (float) -k / 2);
		float v2 = MatrixFunctions.pow(MatrixUtil.det(sigma2), -0.5f);
		FloatMatrix v3 = MatrixFunctions.exp(x1.mmul(Solve.pinv(sigma2)).mul(x1).rowSums().mmul(-0.5f));

		return v3.mmul(v1 * v2);
	}

	/**
	 * 查找精度阈值
	 * 
	 * @param yVal
	 * @param pVal
	 * @return
	 */
	public float selectThreshold(FloatMatrix pVal) {

		float pMax = pVal.columnMaxs().get(0);
		float pMin = pVal.columnMins().get(0);
		float stepsize = (pMax - pMin) / 1000;

		float f1Temp;
		float tp, fp, fn, pre, rec;
		FloatMatrix cvPredictions;
		for (float eps = pMin; eps < pMax; eps += stepsize) {

			cvPredictions = pVal.lt(eps);
			tp = cvPredictions.eq(1f).and(yVal.eq(1f)).columnSums().get(0);
			fp = cvPredictions.eq(1f).and(yVal.eq(0f)).columnSums().get(0);
			fn = cvPredictions.eq(0f).and(yVal.eq(1f)).columnSums().get(0);

			pre = tp / (tp + fp + ((tp + fp) == 0 ? 1 : 0));
			rec = tp / (tp + fn + ((tp + fn) == 0 ? 1 : 0));
			f1Temp = 2 * pre * rec / (pre + rec + ((pre + rec) == 0 ? 1 : 0));

			if (f1Temp > f1) {
				f1 = f1Temp;
				epsilon = eps;
			}
		}

		return epsilon;
	}

	/**
	 * 运行异常检测
	 * 
	 * @return outliers 异常值
	 */
	public FloatMatrix run() {

		logger.info("计算交叉验证集多元高斯分布概率...\n");
		FloatMatrix pVal = multivariateGaussian(xVal);

		logger.info("查找异常值精度阈值...\n");
		float epsilon = selectThreshold(pVal);

		logger.info("计算待评估集多元高斯分布概率...\n");
		FloatMatrix p = multivariateGaussian(x);

		logger.info("查找待评估集异常值...\n");
		int[] indices = p.lt(epsilon).findIndices();

		return x.get(indices);
	}

	public float getEpsilon() {
		return epsilon;
	}

	public float getF1() {
		return f1;
	}
}
