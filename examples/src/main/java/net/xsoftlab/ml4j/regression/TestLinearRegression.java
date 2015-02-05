package net.xsoftlab.ml4j.regression;

import java.io.IOException;

import net.xsoftlab.ml4j.util.FeatureNormalize;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class TestLinearRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = System.getProperty("user.dir") + "/resources/linearRegression/house.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("特征标准化...\n");
		FeatureNormalize featureNormalize = new FeatureNormalize(matrixs[0]);
		FloatMatrix norMatrix = featureNormalize.normalize();

		logger.info("添加截距项...\n");
		FloatMatrix x = MatrixUtil.addIntercept(norMatrix);

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(x, matrixs[1], 0.01f, 400);
		FloatMatrix theta = lr.train();

		logger.info("训练完成.\n \t theta = {}\n", theta);

		toc();
	}
}