package net.xsoftlab.ml4j.util;

import java.util.ArrayList;
import java.util.List;

import net.xsoftlab.ml4j.supervised.BaseRegression;

import org.jblas.FloatMatrix;

/**
 * 批量梯度下降
 * 
 * @author 王彦超
 *
 */
public class GradientDescent {

	private boolean flag = true;// 是否记录历记录
	private List<FloatMatrix> history = null;// theta history

	private BaseRegression regression;

	public GradientDescent(BaseRegression regression) {
		super();
		this.regression = regression;
		this.history = new ArrayList<FloatMatrix>();
	}

	/**
	 * 初始化
	 * 
	 * @param flag
	 *            是否记录历史记录
	 */
	public GradientDescent(BaseRegression regression, boolean flag) {
		super();
		this.flag = flag;
		this.regression = regression;
		if (flag) {
			this.history = new ArrayList<FloatMatrix>();
		}
	}

	/**
	 * 计算theta
	 * 
	 * @param theta
	 *            参数
	 * @param iterations
	 *            训练次数
	 * @return theta
	 */
	public FloatMatrix compute(FloatMatrix theta, int iterations) {

		for (int i = 0; i < iterations; i++) {

			theta = theta.sub(regression.computeGradient(theta));

			if (flag)
				history.add(theta);
		}

		return theta;
	}

	public List<FloatMatrix> getHistory() {
		return history;
	}
}
