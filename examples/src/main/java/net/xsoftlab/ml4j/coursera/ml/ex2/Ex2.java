package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.GradientDescent;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.FeatureNormalize;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = RESOURCES_PATH + "/coursera/ml/ex2/ex2data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("训练集特征标准化...\n");
		FeatureNormalize trainNormalize = new FeatureNormalize(matrixs[0], true);
		FloatMatrix x = trainNormalize.normalize();

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(x, matrixs[1]);
		
		logger.info("使用执行训练...\n");
		GradientDescent gd = new GradientDescent(model, 10f, 100);
		FloatMatrix theta = gd.compute();
		
		logger.info("准确度测算...\n");
		x = x.mmul(theta);
		float p = x.ge(0.5f).eq(matrixs[1]).mean() * 100;

		logger.info("训练完成.\n\t theta = {} \n\t准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}