package net.xsoftlab.ml4j.ufldl.newVersion.ex1;

import java.io.IOException;
import java.util.Map;

import net.xsoftlab.ml4j.minfunc.BFGS;
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

		// logger.info("使用梯度下降执行训练...\n");
		// GradientDescent gd = new GradientDescent(model, 100f);
		// FloatMatrix theta = gd.compute();

		logger.info("使用BFGS执行训练...\n");
		BFGS bfgs = new BFGS(model);
		FloatMatrix theta = bfgs.compute();

		logger.info("准确度测算...\n");
		float p = MatrixUtil.sigmoid(train[0].mmul(theta)).ge(0.5f).eq(train[1]).mean() * 100;
		float p1 = MatrixUtil.sigmoid(test[0].mmul(theta)).ge(0.5f).eq(test[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t 训练集准确度 = {}% \n\t 测试集准确度 = {}%", new Object[] { theta, p, p1 });

		toc();
	}
}