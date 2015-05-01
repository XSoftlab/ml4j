package net.xsoftlab.ml4j.coursera.ml.ex4;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.NeuralNetworks;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex4_NeuralNetworks extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String x_path = COURSE_ML_PATH + "/ex3/X.data";
		String y_path = COURSE_ML_PATH + "/ex3/y.data";
		// String theta1_path = COURSE_ML_PATH + "/ex4/Theta1.data";
		// String theta2_path = COURSE_ML_PATH + "/ex4/Theta2.data";

		FloatMatrix X = MatrixUtil.loadData(x_path, "\\s+", true);
		FloatMatrix y = MatrixUtil.loadData(y_path, "\\s+");
		// FloatMatrix theta1 = MatrixUtil.loadData(theta1_path, "\\s+");
		// FloatMatrix theta2 = MatrixUtil.loadData(theta2_path, "\\s+");

		y.put(y.eq(10), 0);// 把10改成0

		logger.info("参数初始化...\n");
		int inputLayerSize = 400; // 20x20 Input Images of Digits
		int hiddenLayerSize = 25; // 25 hidden units
		int numLabels = 10; // 10 labels, from 1 to 10
		// FloatMatrix all_theta = MatrixUtil.merge(theta1.data, theta2.data);

		logger.info("模型初始化...\n");
		BaseModel model = new NeuralNetworks(inputLayerSize, hiddenLayerSize, numLabels, X, y, 1);
		// model.compute(all_theta, 1);
		// System.out.println(model.getCost());

		logger.info("执行训练...\n");
		MinFunc minFunc = new LBFGS(model);
		FloatMatrix theta = minFunc.train();

		logger.info("准确度测算...\n");
		float p = model.evaluate(theta);

		logger.info("训练完成.\n\t 准确度 = {}% \n", new Object[] { p });
		toc();
	}
}