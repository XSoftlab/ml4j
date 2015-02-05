package net.xsoftlab.ml4j.coursera.ml;

import java.io.IOException;

import net.xsoftlab.ml4j.common.DataLoader;
import net.xsoftlab.ml4j.common.FeatureNormalize;
import net.xsoftlab.ml4j.regression.LinearRegression;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex1_multi extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/coursera/ml/ex1/ex1data2.txt";
		FloatMatrix[] matrixs = DataLoader.loadDataWithXY(path, ",", false);

		logger.info("特征标准化...\n");
		FeatureNormalize featureNormalize = new FeatureNormalize(matrixs[0]);
		FloatMatrix norMatrix = featureNormalize.normalize();

		logger.info("添加截距项...\n");
		FloatMatrix x = new FloatMatrix(norMatrix.rows, norMatrix.columns + 1);
		x.putColumn(0, FloatMatrix.ones(norMatrix.rows));
		for (int i = 0; i < norMatrix.columns; i++) {
			x.putColumn(i + 1, norMatrix.getColumn(i));
		}

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(x, matrixs[1], 0.01f, 400);
		FloatMatrix theta = lr.train();

		logger.info("训练完成.\n \t theta = {}\n", theta);

		toc();
	}
}