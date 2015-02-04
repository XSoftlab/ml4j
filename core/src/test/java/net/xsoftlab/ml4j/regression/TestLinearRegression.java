package net.xsoftlab.ml4j.regression;

import java.io.IOException;

import net.xsoftlab.ml4j.common.DataLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class TestLinearRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		System.out.println("加载数据...\n");

		String path = System.getProperty("user.dir") + "/resources/linearRegression/house.txt";
		FloatMatrix[] matrix = DataLoader.loadDataWithXY(path, ",", true);
		FloatMatrix theta = FloatMatrix.zeros(3, 1);// 初始化theta

		System.out.println("训练...\n");
		LinearRegression lr = new LinearRegression(matrix[0], matrix[1], theta, 0.01f, 1500);
		theta = lr.train();

		System.out.println("Theta found by gradient descent: " + theta.get(0) + "," + theta.get(1));

		toc();
	}
}