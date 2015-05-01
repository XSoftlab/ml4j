package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.common.FeatureNormalize;
import net.xsoftlab.ml4j.minfunc.GradientDescent;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2 extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String path = COURSE_ML_PATH + "/ex2/ex2data1.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("训练集特征标准化...\n");
		FeatureNormalize trainNormalize = new FeatureNormalize(matrixs[0], true);
		FloatMatrix x = trainNormalize.normalize();

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(x, matrixs[1]);

		logger.info("执行训练...\n");
		// MinFunc minFunc = new BFGS(model);
		MinFunc minFunc = new GradientDescent(model, 10f, 100);
		FloatMatrix theta = minFunc.train();

		logger.info("准确度测算...\n");
		float p = model.evaluate(theta);

		logger.info("训练完成.\n\t theta = {} \n\t准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}