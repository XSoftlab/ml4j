package net.xsoftlab.ml4j.common;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LinearRegression;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.MnistLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

/**
 * 测试梯度校验
 * 
 * @author 王彦超
 *
 */
public class Test_GradientChecking extends TestUtil {

	public static void main(String[] args) throws IOException {

		TestLinearRegression();
		
		TestLogisticRegression();
	}

	public static void TestLinearRegression() throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = RESOURCES_PATH + "/ufldl/newVersion/ex1/housing.data";
		FloatMatrix matrix = MatrixUtil.loadData(path, "\\s+", false);

		logger.info("打乱数据...\n");
		matrix = MatrixUtil.shuffle(matrix);

		logger.info("分成训练集和测试集...\n");
		FloatMatrix train = matrix.getRange(0, 400, 0, matrix.columns);
		FloatMatrix train_x = train.getRange(0, train.rows, 0, train.columns - 1);
		FloatMatrix train_y = train.getRange(0, train.rows, train.columns - 1, train.columns);

		logger.info("训练集特征标准化...\n");
		FeatureNormalize trainNormalize = new FeatureNormalize(train_x, true);
		train_x = trainNormalize.normalize();

		logger.info("模型初始化...\n");
		BaseModel model = new LinearRegression(train_x, train_y);

		logger.info("执行梯度校验...\n");
		GradientChecking gc = new GradientChecking(model);
		gc.check();

		toc();
	}

	public static void TestLogisticRegression() throws IOException {

		tic();

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = MnistLoader.load(true);
		FloatMatrix[] train = map.get("train");

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(train[0], train[1]);

		logger.info("执行梯度校验...\n");
		GradientChecking gc = new GradientChecking(model);
		gc.check();

		toc();
	}

}