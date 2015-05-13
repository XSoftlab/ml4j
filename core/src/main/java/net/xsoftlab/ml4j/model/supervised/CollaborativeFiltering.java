package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.util.MatrixUtil;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 * 协同过滤
 * 
 * @author X
 *
 * @data 2015年5月8日
 */
public class CollaborativeFiltering extends BaseModel {

	private FloatMatrix r;

	private int rows;
	private int columns;
	private int features;// 特征数

	/**
	 * 初始化
	 * 
	 * @param x
	 * @param theta
	 * @param y
	 * @param r
	 */
	public CollaborativeFiltering(FloatMatrix y, FloatMatrix r, int features) {
		super();
		this.y = y;
		this.r = r;

		this.rows = y.rows;
		this.columns = y.columns;
		this.features = features;
	}

	/**
	 * 初始化
	 * 
	 * @param x
	 * @param theta
	 * @param y
	 * @param r
	 * @param lambda 正则化系数
	 */
	public CollaborativeFiltering(FloatMatrix y, FloatMatrix r, int features, float lambda) {
		this(y, r, features);

		this.lambda = lambda;
	}

	@Override
	public Object compute(FloatMatrix params, int flag) {

		x = params.getRange(0, rows * features);
		FloatMatrix theta = params.getRange(rows * features, params.length);

		x = x.reshape(rows, features);
		theta = theta.reshape(columns, features);

		if (flag == 1 || flag == 3) {
			FloatMatrix M = MatrixFunctions.pow(x.mmul(theta.transpose()).sub(y), 2);
			this.cost = M.mul(r).columnSums().rowSums().get(0) / 2;

			if (lambda != 0) {
				float cost1 = (lambda / 2)
						* (MatrixFunctions.pow(theta, 2).columnSums().rowSums().get(0) + MatrixFunctions.pow(x, 2)
								.columnSums().rowSums().get(0));
				this.cost += cost1;
			}
		}

		if (flag == 2 || flag == 3) {

			FloatMatrix xGrad = FloatMatrix.zeros(x.rows, x.columns);
			FloatMatrix thetaGrad = FloatMatrix.zeros(theta.rows, theta.columns);

			int[] indices;
			FloatMatrix thetaTemp;
			FloatMatrix xTemp;
			FloatMatrix yTemp;
			for (int i = 0; i < rows; i++) {
				indices = r.getRow(i).eq(1).findIndices();
				if (indices.length == 0)
					continue;

				thetaTemp = theta.getRows(indices);
				yTemp = y.getRow(i).get(indices);
				xGrad.putRow(i, x.getRow(i).mmul(thetaTemp.transpose()).sub(yTemp).mmul(thetaTemp));
			}
			xGrad = xGrad.add(x.mmul(lambda));

			for (int i = 0; i < columns; i++) {
				indices = r.getColumn(i).eq(1).findIndices();
				if (indices.length == 0)
					continue;

				xTemp = x.getRows(indices);
				yTemp = y.getColumn(i).get(indices);
				thetaGrad.putRow(i, xTemp.mmul(theta.getRow(i).transpose()).sub(yTemp).transpose().mmul(xTemp));
			}
			thetaGrad = thetaGrad.add(theta.mmul(lambda));

			this.gradient = MatrixUtil.merge(xGrad.data, thetaGrad.data);
		}

		return flag == 1 ? cost : gradient;
	}

	@Override
	public float evaluate(FloatMatrix theta, FloatMatrix x, FloatMatrix y) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public FloatMatrix getInitTheta() {

		FloatMatrix x = FloatMatrix.rand(rows, features);
		FloatMatrix theta = FloatMatrix.rand(columns, features);
		return MatrixUtil.merge(x.data, theta.data);
	}

	public FloatMatrix normalizeRatings() {

		int[] indices;
		FloatMatrix yMean = FloatMatrix.zeros(rows, 1);
		FloatMatrix yNorm = FloatMatrix.zeros(rows, columns);

		for (int i = 0; i < rows; i++) {
			indices = r.getRow(i).eq(1).findIndices();
			yMean.put(i, y.getRow(i).get(indices).mean());
			yNorm.getRow(i).put(indices, y.getRow(i).get(indices).sub(yMean.get(i)));
		}

		return yMean;
	}
}
