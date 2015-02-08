package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MathUtil;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/coursera/ml/ex2/ex2data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", true);

		logger.info("执行训练...\n");
		LogisticRegression lr = new LogisticRegression(matrixs[0], matrixs[1], 0.001f, 100000);
		FloatMatrix theta = lr.train();

		logger.info("计算均方差...\n");
		float rms = MathUtil.std(matrixs[0].mmul(theta), matrixs[1]);

		logger.info("训练完成.\n \t theta = {} \n\t RMS = {}\n", new Object[] { theta, rms });

		//TODO 随机梯度下降 继续完善ex2 观看coursera视频
		toc();
	}
}