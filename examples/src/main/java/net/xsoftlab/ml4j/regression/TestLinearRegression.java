package net.xsoftlab.ml4j.regression;

import java.io.IOException;

import net.xsoftlab.ml4j.common.DataLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class TestLinearRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/linearRegression/house.txt";
		FloatMatrix[] matrixs = DataLoader.loadDataWithXY(path, ",", true);

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(matrixs[0], matrixs[1], 0.01f, 1500);
		FloatMatrix theta = lr.train();

		logger.info("训练完成.\n \t theta = {}\n", theta);

		toc();
	}
}