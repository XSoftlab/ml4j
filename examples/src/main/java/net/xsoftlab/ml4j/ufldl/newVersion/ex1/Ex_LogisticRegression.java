package net.xsoftlab.ml4j.ufldl.newVersion.ex1;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.MnistLoader;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex_LogisticRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		Map<String, FloatMatrix[]> map = MnistLoader.load(true);
		FloatMatrix[] train = map.get("train");
		FloatMatrix[] test = map.get("test");

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(train[0], train[1]);

		logger.info("执行训练...\n");
		// MinFunc minFunc =new GradientDescent(model, 100f);
		MinFunc minFunc = new LBFGS(model);
		FloatMatrix theta = minFunc.compute();

		logger.info("准确度测算...\n");
		float p = MatrixUtil.sigmoid(train[0].mmul(theta)).ge(0.5f).eq(train[1]).mean() * 100;
		float p1 = MatrixUtil.sigmoid(test[0].mmul(theta)).ge(0.5f).eq(test[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t 训练集准确度 = {}% \n\t 测试集准确度 = {}%", new Object[] { theta, p, p1 });

		// ufldl的测试集准确度为100%，而此处为99.95272%
		// 是因为ufldl的程序里进行了%2.1f的输出控制，进行了四舍五入。
		toc();
	}
}