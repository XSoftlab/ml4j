package net.xsoftlab.ml4j.coursera.ml.ex1;

import java.io.IOException;

import net.xsoftlab.ml4j.supervised.LinearRegression;
import net.xsoftlab.ml4j.util.MathUtil;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex1 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/coursera/ml/ex1/ex1data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", true);

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(matrixs[0], matrixs[1], 0.01f, 1500);
		FloatMatrix theta = lr.train();

		logger.info("计算均方差...\n");
		float rms = MathUtil.std(matrixs[0].mmul(theta), matrixs[1]);

		logger.info("训练完成.\n \t theta = {} , RMS = {}\n", new Object[] { theta, rms });

		toc();
	}
}