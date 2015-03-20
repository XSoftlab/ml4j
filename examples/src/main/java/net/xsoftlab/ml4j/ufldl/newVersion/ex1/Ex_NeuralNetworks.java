package net.xsoftlab.ml4j.ufldl.newVersion.ex1;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.NeuralNetworks;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.MnistLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex_NeuralNetworks extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = MnistLoader.load(false);
		FloatMatrix[] train = map.get("train");
		FloatMatrix[] test = map.get("test");

		logger.info("参数初始化...\n");
		int inputLayerSize = 784; // 28x28 Input Images of Digits
		int hiddenLayerSize = 25; // 25 hidden units
		int numLabels = 10; // 10 labels, from 1 to 10

		logger.info("模型初始化...\n");
		BaseModel model = new NeuralNetworks(inputLayerSize, hiddenLayerSize, numLabels, train[0], train[1], 1);

		logger.info("执行训练...\n");
		MinFunc minFunc = new LBFGS(model);
		FloatMatrix theta = minFunc.compute();

		FloatMatrix theta1 = theta.getRange(0, (inputLayerSize + 1) * hiddenLayerSize);
		FloatMatrix theta2 = theta.getRange((inputLayerSize + 1) * hiddenLayerSize, theta.length);
		theta1 = theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
		theta2 = theta2.reshape(numLabels, hiddenLayerSize + 1);

		logger.info("准确度测算...\n");
		FloatMatrix h1 = MatrixUtil.sigmoid(train[0].mmul(theta1.transpose()));
		h1 = MatrixUtil.addIntercept(h1);
		FloatMatrix h2 = MatrixUtil.sigmoid(h1.mmul(theta2.transpose()));
		int[] index = h2.rowArgmaxs();
		float[] pred = new float[index.length];

		for (int i = 0; i < index.length; i++)
			pred[i] = index[i];

		float p = train[1].eq(new FloatMatrix(pred)).mean() * 100;

		h1 = MatrixUtil.sigmoid(test[0].mmul(theta1.transpose()));
		h1 = MatrixUtil.addIntercept(h1);
		h2 = MatrixUtil.sigmoid(h1.mmul(theta2.transpose()));
		index = h2.rowArgmaxs();
		pred = new float[index.length];

		for (int i = 0; i < index.length; i++)
			pred[i] = index[i];

		float p1 = test[1].eq(new FloatMatrix(pred)).mean() * 100;

		logger.info("训练完成.\n\t 训练集准确度 = {}% \n\t 测试集准确度 = {}% \n", new Object[] { p, p1 });
		toc();
	}
}