package net.xsoftlab.ml4j.model.unsupervised;

import org.jblas.FloatMatrix;
import org.jblas.Singular;

/**
 * PCA
 * 
 * @author X
 *
 * @data 2015年5月6日
 */
public class PCA {

	private FloatMatrix u;// eigenvectors 0
	private FloatMatrix s;// eigenvectors 1

	private FloatMatrix x;// 特征矩阵
	private int m;// 样本数量

	/**
	 * 初始化PCA
	 * 
	 * @param x 特征矩阵
	 */
	public PCA(FloatMatrix x) {
		super();
		this.x = x;

		this.m = x.rows;
		FloatMatrix Sigma = x.transpose().mmul(x).div(m);
		FloatMatrix[] result = Singular.fullSVD(Sigma);

		this.u = result[0].neg();// 与matlab保持一致
		this.s = result[1];
	}

	/**
	 * PROJECTDATA Computes the reduced data representation when projecting only
	 * on to the top k eigenvectors
	 * Z = projectData(K) computes the projection of
	 * the normalized inputs X into the reduced dimensional space spanned by
	 * the first K columns of U. It returns the projected examples in Z.
	 * 
	 * @param k the first K columns of U
	 * @return the projected examples
	 */
	public FloatMatrix projectData(int k) {

		FloatMatrix uReduce = u.getRange(0, u.rows, 0, k);
		return x.mmul(uReduce);
	}

	/**
	 * RECOVERDATA Recovers an approximation of the original data when using the
	 * projected data
	 * 
	 * @param z projected data
	 * @param k the first K columns of U
	 * @return
	 */
	public FloatMatrix recoverData(FloatMatrix z, int k) {
		FloatMatrix uReduce = u.getRange(0, u.rows, 0, k);
		return z.mmul(uReduce.transpose());
	}

	public FloatMatrix getU() {
		return u;
	}

	public void setU(FloatMatrix u) {
		this.u = u;
	}

	public FloatMatrix getS() {
		return s;
	}

	public void setS(FloatMatrix s) {
		this.s = s;
	}
}
