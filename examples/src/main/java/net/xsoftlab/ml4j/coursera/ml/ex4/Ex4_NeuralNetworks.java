package net.xsoftlab.ml4j.coursera.ml.ex4;

import java.io.IOException;

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
		String theta1_path = RESOURCES_PATH + "/coursera/ml/ex4/Theta1.data";
		String theta2_path = RESOURCES_PATH + "/coursera/ml/ex4/Theta2.data";

		FloatMatrix X = MatrixUtil.loadData(x_path, "\\s+", true);
		FloatMatrix y = MatrixUtil.loadData(y_path, "\\s+");
		FloatMatrix theta1 = MatrixUtil.loadData(theta1_path, "\\s+");
		FloatMatrix theta2 = MatrixUtil.loadData(theta2_path, "\\s+");

		logger.info("参数初始化...\n");

		int input_layer_size = 400; // 20x20 Input Images of Digits
		int hidden_layer_size = 25; // 25 hidden units
		// (note that we have mapped "0" to label 10)
		int num_labels = 10; // 10 labels, from 1 to 10
		FloatMatrix all_theta = MatrixUtil.merge(theta1.data, theta2.data);

		logger.info("模型 初始化...\n");

		BaseModel model = new NeuralNetworks(input_layer_size, hidden_layer_size, num_labels, X, y, 1);
		System.out.println("cost:" + model.cost(all_theta));

		toc();
	}
}