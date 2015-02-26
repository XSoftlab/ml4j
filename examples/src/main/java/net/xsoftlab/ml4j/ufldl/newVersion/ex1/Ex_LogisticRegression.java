package net.xsoftlab.ml4j.ufldl.newVersion.ex1;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex_LogisticRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = LoadMnist.load(true);
		FloatMatrix[] train = map.get("train");
		// FloatMatrix[] test = map.get("test");

		logger.info("执行训练...\n");
		LogisticRegression lr = new LogisticRegression(train[0], train[1], 0.01f, 400, 0);
		lr.setPrintCost(false);
		FloatMatrix theta = lr.train();

		logger.info("准确度测算...\n");
		float p = MatrixUtil.sigmoid(train[0].mmul(theta)).ge(0.5f).eq(train[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t 准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}