package net.xsoftlab.ml4j.coursera.ml.ex8;

import java.io.IOException;

import net.xsoftlab.ml4j.model.unsupervised.AnomalyDetection;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex8_AnomalyDetection extends TestUtil {

	public static void part1() throws IOException {

		logger.info("加载数据...\n");
		String path_x = COURSE_ML_PATH + "/ex8/data1_X.data";
		String path_xval = COURSE_ML_PATH + "/ex8/data1_Xval.data";
		String path_yval = COURSE_ML_PATH + "/ex8/data1_yval.data";

		// 待评估集
		FloatMatrix X = MatrixUtil.loadData(path_x, "\\s+");
		// 交叉验证集 - 用于异常精度值评估
		FloatMatrix xVal = MatrixUtil.loadData(path_xval, "\\s+");
		FloatMatrix yVal = MatrixUtil.loadData(path_yval, "\\s+");

		logger.info("初始化...\n");
		AnomalyDetection ad = new AnomalyDetection(X, xVal, yVal);

		logger.info("运行AnomalyDetection...\n");
		FloatMatrix outliers = ad.run();

		logger.info("结果如下：{}", outliers);
	}

	public static void part2() throws IOException {

		logger.info("加载数据...\n");
		String path_x = COURSE_ML_PATH + "/ex8/data2_X.data";
		String path_xval = COURSE_ML_PATH + "/ex8/data2_Xval.data";
		String path_yval = COURSE_ML_PATH + "/ex8/data2_yval.data";

		// 待评估集
		FloatMatrix X = MatrixUtil.loadData(path_x, "\\s+");
		// 交叉验证集 - 用于异常精度值评估
		FloatMatrix xVal = MatrixUtil.loadData(path_xval, "\\s+");
		FloatMatrix yVal = MatrixUtil.loadData(path_yval, "\\s+");

		logger.info("初始化...\n");
		AnomalyDetection ad = new AnomalyDetection(X, xVal, yVal);

		logger.info("运行AnomalyDetection...\n");
		FloatMatrix outliers = ad.run();

		logger.info("Best epsilon found using cross-validation: {}\n", ad.getEpsilon());
		logger.info("Best F1 on Cross Validation Set:  {}\n", ad.getF1());
		logger.info("# Outliers found: {}\n", outliers.length);
		logger.info("   (you should see a value epsilon of about 1.38e-18)");
	}

	public static void main(String[] args) throws IOException {
		// part1();
		part2();
	}
}
