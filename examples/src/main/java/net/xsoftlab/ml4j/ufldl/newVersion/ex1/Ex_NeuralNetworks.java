package net.xsoftlab.ml4j.ufldl.newVersion.ex1;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.NeuralNetworks;
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
		FloatMatrix theta = minFunc.train();

		logger.info("准确度测算...\n");

		float p = model.evaluate(theta);
		float p1 = model.evaluate(theta, test[0], test[1]);

		logger.info("训练完成.\n\t 训练集准确度 = {}% \n\t 测试集准确度 = {}% \n", new Object[] { p, p1 });
		toc();
	}
}