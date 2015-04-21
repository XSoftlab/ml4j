package net.xsoftlab.ml4j.common;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.MnistLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

/**
 * 测试保存和加载Theta
 * 
 * @author 王彦超
 *
 */
public class Test_SaveAndLoadTheta extends TestUtil {

	public static void main(String[] args) throws IOException {

		String path = "d:/data/theta.data";

		tic();

		saveTheta(path);
		loadTheta(path);

		toc();
	}

	public static void loadTheta(String path) throws IOException {

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = MnistLoader.load(true);
		FloatMatrix[] train = map.get("train");
		FloatMatrix[] test = map.get("test");

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(train[0], train[1]);

		logger.info("加载 theta...\n");
		FloatMatrix theta = new FloatMatrix();
		theta.load(path);

		logger.info("准确度测算...\n");
		float p = model.evaluate(theta);
		float p1 = model.evaluate(theta, test[0], test[1]);

		logger.info("训练集准确度 = {}% \n\t 测试集准确度 = {}%", new Object[] { p, p1 });
	}

	public static void saveTheta(String path) throws IOException {

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = MnistLoader.load(true);
		FloatMatrix[] train = map.get("train");
		FloatMatrix[] test = map.get("test");

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(train[0], train[1]);

		logger.info("执行训练...\n");
		MinFunc minFunc = new LBFGS(model);
		FloatMatrix theta = minFunc.train();

		logger.info("保存数据...\n");
		MatrixUtil.save(theta, path);

		logger.info("准确度测算...\n");
		float p = model.evaluate(theta);
		float p1 = model.evaluate(theta, test[0], test[1]);

		logger.info("训练完成.\n\t theta = {} \n\t 训练集准确度 = {}% \n\t 测试集准确度 = {}%", new Object[] { theta, p, p1 });
	}
}