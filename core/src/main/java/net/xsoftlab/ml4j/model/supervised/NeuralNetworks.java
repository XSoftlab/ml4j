package net.xsoftlab.ml4j.model.supervised;

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

	public NeuralNetworks() {
		super();
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
	 *            特征矩阵
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
	 *            特征矩阵
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
	public void compute(FloatMatrix theta, int flag) {

		FloatMatrix theta1 = theta.getRange(0, (inputLayerSize + 1) * hiddenLayerSize);
		FloatMatrix theta2 = theta.getRange((inputLayerSize + 1) * hiddenLayerSize, theta.length);

		theta1 = theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
		theta2 = theta2.reshape(numLabels, hiddenLayerSize + 1);

		FloatMatrix a1 = x;
		FloatMatrix z2 = a1.mmul(theta1.transpose());
		FloatMatrix a2 = MatrixUtil.addIntercept(MatrixUtil.sigmoid(z2));
		FloatMatrix a3 = MatrixUtil.sigmoid(a2.mmul(theta2.transpose()));

		FloatMatrix Y = FloatMatrix.zeros(a3.rows, a3.columns);
		for (int i = 0; i < numLabels; i++)
			Y.putColumn(i, y.eq(i));

		if (flag == 1 || flag == 3) {
			// -Y .* log(a3)
			FloatMatrix h1 = Y.neg().mul(MatrixUtil.log(a3));
			// (1 - Y) .* log(1 - a3)
			FloatMatrix h2 = (Y.neg().add(1f)).mul(MatrixUtil.log(a3.neg().add(1f)));

			this.cost = 1f / m * h1.sub(h2).sum();// 1 / m * (h1 - h2)

			if (lambda != 0) {
				FloatMatrix theta3 = theta1.getRange(1, theta1.length);
				FloatMatrix theta4 = theta2.getRange(1, theta2.length);
				float cost1 = lambda / (2 * m)
						* (theta3.transpose().mmul(theta3).sum() + theta4.transpose().mmul(theta4).sum());
				cost += cost1;
			}
		}

		if (flag == 2 || flag == 3) {

			FloatMatrix l3 = a3.sub(Y);
			z2 = MatrixUtil.addIntercept(z2);
			FloatMatrix l2 = l3.mmul(theta2).mul(sigmoidGradient(z2));

			l2 = l2.getRange(0, l2.rows, 1, l2.columns);// l2(:,2:end)
			FloatMatrix theta1Grad = l2.transpose().mmul(a1).add(theta1.mul(lambda));
			FloatMatrix theta2Grad = l3.transpose().mmul(a2).add(theta2.mul(lambda));
			if (lambda != 0) {
				FloatMatrix theta1Grad1 = l2.transpose().mmul(a1.getColumn(0));
				theta1Grad.putColumn(0, theta1Grad1);

				FloatMatrix theta2Grad1 = l3.transpose().mmul(a2.getColumn(0));
				theta2Grad.putColumn(0, theta2Grad1);
			}

			theta1Grad = theta1Grad.div(m);
			theta2Grad = theta2Grad.div(m);
			this.gradient = MatrixUtil.merge(theta1Grad.data, theta2Grad.data);
		}
	}

	@Override
	public float evaluate(FloatMatrix theta) {

		return evaluate(theta, x, y);
	}

	@Override
	public float evaluate(FloatMatrix theta, FloatMatrix x, FloatMatrix y) {

		FloatMatrix theta1 = theta.getRange(0, (inputLayerSize + 1) * hiddenLayerSize);
		FloatMatrix theta2 = theta.getRange((inputLayerSize + 1) * hiddenLayerSize, theta.length);
		theta1 = theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
		theta2 = theta2.reshape(numLabels, hiddenLayerSize + 1);

		FloatMatrix h1 = MatrixUtil.sigmoid(x.mmul(theta1.transpose()));
		h1 = MatrixUtil.addIntercept(h1);
		FloatMatrix h2 = MatrixUtil.sigmoid(h1.mmul(theta2.transpose()));
		int[] index = h2.rowArgmaxs();
		float[] pred = new float[index.length];

		for (int i = 0; i < index.length; i++)
			pred[i] = index[i];

		float p = y.eq(new FloatMatrix(pred)).mean() * 100;

		return p;
	}

	@Override
	public FloatMatrix getInitTheta() {

		FloatMatrix initialTheta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
		FloatMatrix initialTheta2 = randInitializeWeights(hiddenLayerSize, numLabels);
		return MatrixUtil.merge(initialTheta1.data, initialTheta2.data);
	}

	/**
	 * computes the gradient of the sigmoid function
	 * 
	 * @param z
	 *            evaluated at z
	 * @return sigmoidGradient
	 */
	public FloatMatrix sigmoidGradient(FloatMatrix z) {

		FloatMatrix matrix = MatrixUtil.sigmoid(z);
		return matrix.mul(matrix.neg().add(1f));
	}

	/**
	 * 随机初始化theta
	 * 
	 * @param LIn
	 *            输入单元数量
	 * @param LOut
	 *            输出单元数量
	 * @return theta
	 */
	public FloatMatrix randInitializeWeights(int LIn, int LOut) {

		float epsilonInit = (float) (Math.sqrt(6) / Math.sqrt(LIn + LOut));
		// LIn + 1：添加偏置单元
		return FloatMatrix.rand(LOut, LIn + 1).mul(2f * epsilonInit).sub(epsilonInit);
	}
}
