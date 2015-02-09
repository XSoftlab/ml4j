package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2_reg extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/coursera/ml/ex2/ex2data2.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", true);

		logger.info("执行训练...\n");
		LogisticRegression lr = new LogisticRegression(matrixs[0], matrixs[1], 3f, 1000, 0);
		FloatMatrix theta = lr.train();

		logger.info("准确度测算...\n");
		FloatMatrix x = matrixs[0].mmul(new FloatMatrix(new float[] { -0.1125f, -0.0807f, -0.3491f }));
		float p = x.ge(0.5f).eq(matrixs[1]).mean() * 100;
		System.out.println(x);

		logger.info("训练完成.\n\t theta = {} \n\t 准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}