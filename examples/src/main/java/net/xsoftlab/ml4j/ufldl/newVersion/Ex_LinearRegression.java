package net.xsoftlab.ml4j.ufldl.newVersion;

import java.io.IOException;

import net.xsoftlab.ml4j.regression.LinearRegression;
import net.xsoftlab.ml4j.util.FeatureNormalize;
import net.xsoftlab.ml4j.util.MathUtil;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex_LinearRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = System.getProperty("user.dir") + "/resources/ufldl/newVersion/housing.data";
		FloatMatrix matrixs = MatrixUtil.loadData(path, "\\s+", false);

		logger.info("打乱数据...\n");
		matrixs = MatrixUtil.shuffle(matrixs);

		logger.info("分成训练集和测试集...\n");
		FloatMatrix train = matrixs.getRange(0, 400, 0, matrixs.columns);
		FloatMatrix train_x = train.getRange(0, train.rows, 0, train.columns - 1);
		FloatMatrix train_y = train.getRange(0, train.rows, train.columns - 1, train.columns);

		FloatMatrix test = matrixs.getRange(400, matrixs.rows, 0, matrixs.columns);
		FloatMatrix test_x = train.getRange(0, test.rows, 0, test.columns - 1);
		FloatMatrix test_y = train.getRange(0, test.rows, test.columns - 1, test.columns);

		logger.info("训练集特征标准化...\n");
		FeatureNormalize trainNormalize = new FeatureNormalize(train_x, true);
		train_x = trainNormalize.normalize();

		logger.info("测试集特征标准化...\n");
		FeatureNormalize testNormalize = new FeatureNormalize(test_x, trainNormalize.getMu(),
				trainNormalize.getSigma(), true);
		test_x = testNormalize.normalize();

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(train_x, train_y, 0.3f, 200);
		FloatMatrix theta = lr.train();

		logger.info("计算训练集均方差...\n");
		float trainRms = MathUtil.std(train_x.mmul(theta), train_y);

		logger.info("计算测试集均方差...\n");
		float testRms = MathUtil.std(test_x.mmul(theta), test_y);

		logger.info("训练完成.\n\t theta = {} \n\t trainRms = {}\n\t testRms = {}\n", new Object[] { theta, trainRms,
				testRms });

		toc();
	}
}