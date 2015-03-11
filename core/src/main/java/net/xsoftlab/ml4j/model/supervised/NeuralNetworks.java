package net.xsoftlab.ml4j.model.supervised;

import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.util.MatrixUtil;

import org.jblas.FloatMatrix;

/**
 * 神经网络模型 - 单隐层 - 分类
 * 
 * @author 王彦超
 * 
 */
public class NeuralNetworks extends BaseModel {

	private int inputLayerSize;// 输入单元大小
	private int hiddenLayerSize;// 隐藏单元大小
	private int numLabels;// 标签个数

	/**
	 * 初始化
	 * 
	 * @param inputLayerSize
	 *            输入单元大小
	 * @param hiddenLayerSize
	 *            隐藏单元大小
	 * @param numLabels
	 *            标签个数
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 */
	public NeuralNetworks(int inputLayerSize, int hiddenLayerSize, int numLabels, FloatMatrix x, FloatMatrix y) {
		super();
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numLabels = numLabels;
		this.x = x;
		this.y = y;

		this.m = y.length;
	}

	/**
	 * 初始化
	 * 
	 * @param inputLayerSize
	 *            输入单元大小
	 * @param hiddenLayerSize
	 *            隐藏单元大小
	 * @param numLabels
	 *            标签个数
	 * @param x
	 *            特征值
	 * @param y
	 *            标签
	 * @param lambda
	 *            正则系数
	 */
	public NeuralNetworks(int inputLayerSize, int hiddenLayerSize, int numLabels, FloatMatrix x, FloatMatrix y,
			float lambda) {
		this(inputLayerSize, hiddenLayerSize, numLabels, x, y);

		this.lambda = lambda;
	}

	@Override
	public FloatMatrix function(FloatMatrix x, FloatMatrix theta) {

		return MatrixUtil.sigmoid(x.mmul(theta));
	}

	@Override
	public FloatMatrix gradient(FloatMatrix theta) {

		// sigmoid(X * theta) - y
		FloatMatrix h = function(x, theta).sub(y);
		// x' * h * (alpha / m)
		FloatMatrix h1 = x.transpose().mmul(h);
		FloatMatrix h2 = h1.add(theta.mul(lambda));

		if (lambda != 0) {
			FloatMatrix h3 = x.getColumn(0).transpose().mmul(h);
			h2.put(0, h3.get(0));
		}

		return h2.div(m);
	}

	@Override
	public float cost(FloatMatrix theta) {

		FloatMatrix theta1 = theta.getRange(0, (inputLayerSize + 1) * hiddenLayerSize);
		FloatMatrix theta2 = theta.getRange((inputLayerSize + 1) * hiddenLayerSize, theta.length);

		theta1 = theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
		theta2 = theta2.reshape(numLabels, hiddenLayerSize + 1);

		FloatMatrix a2 = MatrixUtil.addIntercept(function(x, theta1.transpose()));
		FloatMatrix a3 = function(a2, theta2.transpose());

		FloatMatrix Y = FloatMatrix.zeros(a3.rows, a3.columns);
		for (int i = 0; i < numLabels; i++)
			Y.putColumn(i, y.eq(i + 1));

		// -Y .* log(a3)
		FloatMatrix h1 = Y.neg().mul(MatrixUtil.log(a3));
		// (1 - Y) .* log(1 - a3)
		FloatMatrix h2 = (Y.neg().add(1f)).mul(MatrixUtil.log(a3.neg().add(1f)));

		float cost = 1f / m * h1.sub(h2).sum();// 1 / m * (h1 - h2)

		if (lambda != 0) {
			FloatMatrix theta3 = theta1.getRange(1, theta1.length);
			FloatMatrix theta4 = theta2.getRange(1, theta2.length);
			float cost1 = lambda / (2 * m)
					* (theta3.transpose().mmul(theta3).sum() + theta4.transpose().mmul(theta4).sum());
			cost += cost1;
		}

		return cost;
	}
}
