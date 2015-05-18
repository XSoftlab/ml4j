package net.xsoftlab.ml4j.coursera.ml.ex1;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.GradientDescent;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LinearRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex1 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = COURSE_ML_PATH + "/ex1/ex1data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", true);

		logger.info("模型初始化...\n");
		BaseModel model = new LinearRegression(matrixs[0], matrixs[1]);

		logger.info("执行训练...\n");
		MinFunc minFunc = new GradientDescent(model, 0.024f, 1500);
		FloatMatrix theta = model.train(minFunc);

		logger.info("计算均方差...\n");
		float rms = model.evaluate(theta);

		logger.info("训练完成.\n \t theta = {} \n\t RMS = {}\n", new Object[] { theta, rms });

		toc();
	}
}