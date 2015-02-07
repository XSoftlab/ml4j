package net.xsoftlab.ml4j.util;

import org.jblas.FloatMatrix;

/**
 * 特征标准化
 * 
 * @author 王彦超
 *
 */
public class FeatureNormalize {

	private FloatMatrix data;
	private FloatMatrix data_normal;// 标准化后的结果
	private FloatMatrix mu;// 平均值
	private FloatMatrix sigma;// 标准差
	private FloatMatrix temp;
	private boolean intercept = true;

	/**
	 * 初始化
	 * 
	 * @param data
	 *            要标准化的数据
	 */
	public FeatureNormalize(FloatMatrix data) {

		this(data, true);
	}

	/**
	 * 初始化
	 * 
	 * @param data
	 *            要标准化的数据
	 * @param intercept
	 *            是否添加截距项
	 */
	public FeatureNormalize(FloatMatrix data, boolean intercept) {
		super();
		this.data = data;
		this.intercept = intercept;

		mu = data.columnMeans();
		sigma = MathUtil.std(data, 1);
		if (intercept)
			data_normal = new FloatMatrix(data.rows, data.columns + 1);
		else
			data_normal = new FloatMatrix(data.rows, data.columns);
	}

	/**
	 * 初始化
	 * 
	 * @param data
	 *            要标准化的数据
	 * @param intercept
	 *            是否添加截距项
	 */
	public FeatureNormalize(FloatMatrix data, FloatMatrix mu, FloatMatrix sigma, boolean intercept) {
		super();
		this.data = data;
		this.mu = mu;
		this.sigma = sigma;
		this.intercept = intercept;

		if (intercept)
			data_normal = new FloatMatrix(data.rows, data.columns + 1);
		else
			data_normal = new FloatMatrix(data.rows, data.columns);
	}

	/**
	 * 执行标准化
	 * 
	 * @return FloatMatrix
	 */
	public FloatMatrix normalize() {

		if (intercept) {
			data_normal.putColumn(0, FloatMatrix.ones(data_normal.rows));
			for (int i = 0; i < data.columns; i++) {
				temp = data.getColumn(i).sub(mu.get(i));// data[i] - mu[i]
				data_normal.putColumn(i + 1, temp.div(sigma.get(i)));
			}
		} else {
			for (int i = 0; i < data.columns; i++) {
				temp = data.getColumn(i).sub(mu.get(i));// data[i] - mu[i]
				data_normal.putColumn(i, temp.div(sigma.get(i)));
			}
		}

		return data_normal;
	}

	public FloatMatrix getData_normal() {
		return data_normal;
	}

	public FloatMatrix getMu() {
		return mu;
	}

	public FloatMatrix getSigma() {
		return sigma;
	}

}
