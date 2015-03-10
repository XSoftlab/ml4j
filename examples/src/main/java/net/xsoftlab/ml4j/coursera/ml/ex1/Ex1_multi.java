package net.xsoftlab.ml4j.coursera.ml.ex1;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.BFGS;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LinearRegression;
import net.xsoftlab.ml4j.util.FeatureNormalize;
import net.xsoftlab.ml4j.util.MathUtil;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex1_multi extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = RESOURCES_PATH + "/coursera/ml/ex1/ex1data2.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("特征标准化...\n");
		FeatureNormalize featureNormalize = new FeatureNormalize(matrixs[0], true);
		FloatMatrix x = featureNormalize.normalize();

		logger.info("模型初始化...\n");
		BaseModel model = new LinearRegression(x, matrixs[1]);

		// logger.info("使用梯度下降执行训练...\n");
		// GradientDescent gd = new GradientDescent(model, 0.1f);
		// FloatMatrix theta = gd.compute();

		logger.info("使用BFGS执行训练...\n");
		BFGS bfgs = new BFGS(model);
		FloatMatrix theta = bfgs.compute();

		logger.info("计算均方差...\n");
		float rms = MathUtil.std(x.mmul(theta), matrixs[1]);

		logger.info("训练完成.\n\t theta = {} \n\t RMS = {}\n", new Object[] { theta, rms });

		toc();
	}
}