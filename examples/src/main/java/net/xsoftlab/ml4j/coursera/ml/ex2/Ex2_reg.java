package net.xsoftlab.ml4j.coursera.ml.ex2;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.GradientDescent;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex2_reg extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = COURSE_ML_PATH + "/ex2/ex2data2.txt";
		FloatMatrix[] matrixs = MatrixUtil.loadDataWithXY(path, ",", false);

		logger.info("参数处理...\n");
		FloatMatrix x = MapFeature.mapFeature(matrixs[0].getColumn(0), matrixs[0].getColumn(1));

		logger.info("模型初始化...\n");
		BaseModel model = new LogisticRegression(x, matrixs[1], 1);

		logger.info("使用BFGS执行训练...\n");
		// MinFunc minFunc = new BFGS(model);
		MinFunc minFunc = new GradientDescent(model, 1f);
		FloatMatrix theta = minFunc.train();

		logger.info("准确度测算...\n");
		float p = model.evaluate(theta);

		logger.info("训练完成.\n\t theta = {} \n\t 准确度 = {}%", new Object[] { theta, p });

		toc();
	}
}