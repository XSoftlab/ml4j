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

	/**
	 * 初始化
	 * 
	 * @param data
	 *            要标准化的数据
	 */
	public FeatureNormalize(FloatMatrix data) {
		super();
		this.data = data;

		mu = data.columnMeans();
		sigma = data.columnMaxs().sub(data.columnMins());
		data_normal = new FloatMatrix(data.rows, data.columns);
	}

	/**
	 * 执行标准化
	 * 
	 * @return FloatMatrix
	 */
	public FloatMatrix normalize() {

		// (data[i] - mu[i]) / sigma[i]
		for (int i = 0; i < data.columns; i++) {
			temp = data.getColumn(i).sub(mu.get(i));// data[i] - mu[i]
			data_normal.putColumn(i, temp.div(sigma.get(i)));
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
