package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.FeatureNormalize;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/coursera/ml/ex2/ex2data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("训练集特征标准化...\n");
		FeatureNormalize trainNormalize = new FeatureNormalize(matrixs[0], true);
		FloatMatrix x = trainNormalize.normalize();

		logger.info("执行训练...\n");
		LogisticRegression lr = new LogisticRegression(x, matrixs[1], 10f, 100);
		FloatMatrix theta = lr.train();

		logger.info("准确度测算...\n");
		x = x.mmul(theta);
		float p = x.ge(0.5f).eq(matrixs[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}