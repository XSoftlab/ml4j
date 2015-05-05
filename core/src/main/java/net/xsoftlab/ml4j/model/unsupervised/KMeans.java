package net.xsoftlab.ml4j.model.unsupervised;

import net.xsoftlab.ml4j.util.MathUtil;

import org.jblas.FloatMatrix;

/**
 * KMeans
 * 
 * @author X
 *
 * @data 2015年5月6日
 */
public class KMeans {

	private int k;// 聚类中心数量
	private FloatMatrix x;// 特征矩阵
	private FloatMatrix centroids;// 聚类中心

	/**
	 * 初始化 K-means
	 * 
	 * @param x 特征矩阵
	 * @param k 聚类中心数量
	 */
	public KMeans(FloatMatrix x, int k) {
		super();
		this.x = x;
		this.k = k;
		int[] rindices = MathUtil.randperm(k);
		this.centroids = x.getRows(rindices);
	}

	/**
	 * 查找最近的聚类中心索引
	 * 
	 * @return 最近的聚类中心索引
	 */
	public FloatMatrix findClosestCentroids() {

		int length = x.rows;
		FloatMatrix indices = new FloatMatrix(length, 1);

		int index = 0;// 最小值索引
		float temp = Float.MAX_VALUE;
		float value;
		FloatMatrix matrix;
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < k; j++) {
				matrix = x.getRow(i).sub(centroids.getRow(j));
				value = matrix.mmul(matrix.transpose()).get(0);
				if (value < temp) {
					temp = value;
					index = j;
				}
			}
			indices.put(i, index);
			temp = Float.MAX_VALUE;
		}

		return indices;
	}

	/**
	 * 计算聚类中心
	 * 
	 * @param indices 最近的聚类中心索引
	 */
	public void computeCentroids(FloatMatrix indices) {

		int[] index;
		FloatMatrix temp;

		for (int i = 0; i < k; i++) {
			index = indices.eq(i).findIndices();
			temp = x.getRows(index).columnSums();
			centroids.putRow(i, temp.div(index.length));
		}
	}

	/**
	 * 运行K-means
	 * 
	 * @param maxIters 迭代次数
	 * @return 聚类中心点
	 */
	public FloatMatrix run(int maxIters) {

		FloatMatrix indices = null;

		for (int i = 0; i < maxIters; i++) {
			indices = findClosestCentroids();
			computeCentroids(indices);
		}

		return centroids;
	}
}
