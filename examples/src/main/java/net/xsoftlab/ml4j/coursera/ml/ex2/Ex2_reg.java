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
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("参数处理...\n");
		FloatMatrix x = MapFeature.mapFeature(matrixs[0].getColumn(0), matrixs[0].getColumn(1));

		logger.info("执行训练...\n");
		LogisticRegression lr = new LogisticRegression(x, matrixs[1], 30f, 10000, 0.0001f);
		FloatMatrix theta = lr.train();

		logger.info("准确度测算...\n");
		float p = x.mmul(theta).ge(0.5f).eq(matrixs[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t 准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}