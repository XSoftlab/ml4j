package net.xsoftlab.ml4j.coursera.ml.ex4;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.GradientDescent;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.NeuralNetworks;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex4_NeuralNetworks extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String x_path = RESOURCES_PATH + "/coursera/ml/ex3/X.data";
		String y_path = RESOURCES_PATH + "/coursera/ml/ex3/y.data";
		// String theta1_path = RESOURCES_PATH + "/coursera/ml/ex4/Theta1.data";
		// String theta2_path = RESOURCES_PATH + "/coursera/ml/ex4/Theta2.data";

		FloatMatrix X = MatrixUtil.loadData(x_path, "\\s+", true);
		FloatMatrix y = MatrixUtil.loadData(y_path, "\\s+");
		// FloatMatrix theta1 = MatrixUtil.loadData(theta1_path, "\\s+");
		// FloatMatrix theta2 = MatrixUtil.loadData(theta2_path, "\\s+");

		logger.info("参数初始化...\n");
		int inputLayerSize = 400; // 20x20 Input Images of Digits
		int hiddenLayerSize = 25; // 25 hidden units
		// (note that we have mapped "0" to label 10)
		int numLabels = 10; // 10 labels, from 1 to 10
		// FloatMatrix all_theta = MatrixUtil.merge(theta1.data, theta2.data);

		logger.info("模型 初始化...\n");
		BaseModel model = new NeuralNetworks(inputLayerSize, hiddenLayerSize, numLabels, X, y, 1);
		// model.compute(all_theta, 1);
		// System.out.println(model.getCost());

		logger.info("使用BFGS执行训练...\n");
		MinFunc minFunc = new GradientDescent(model, 3f);
		FloatMatrix theta = minFunc.compute();

		FloatMatrix theta1 = theta.getRange(0, (inputLayerSize + 1) * hiddenLayerSize);
		FloatMatrix theta2 = theta.getRange((inputLayerSize + 1) * hiddenLayerSize, theta.length);
		theta1 = theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
		theta2 = theta2.reshape(numLabels, hiddenLayerSize + 1);

		logger.info("准确度测算...\n");
		FloatMatrix h1 = MatrixUtil.sigmoid(X.mmul(theta1.transpose()));
		h1 = MatrixUtil.addIntercept(h1);
		FloatMatrix h2 = MatrixUtil.sigmoid(h1.mmul(theta2.transpose()));
		int[] index = h2.rowArgmaxs();
		float[] pred = new float[index.length];

		for (int i = 0; i < index.length; i++)
			pred[i] = index[i] + 1f;

		float p = y.eq(new FloatMatrix(pred)).mean() * 100;

		logger.info("训练完成.\n\t 准确度 = {}% \n", new Object[] { p });
		toc();
	}
}