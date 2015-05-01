package net.xsoftlab.ml4j.coursera.ml.ex3;

import java.io.IOException;

import net.xsoftlab.ml4j.minfunc.LBFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex3_OneVsAll extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");

		String x_path = COURSE_ML_PATH + "/ex3/X.data";
		String y_path = COURSE_ML_PATH + "/ex3/y.data";

		FloatMatrix X = MatrixUtil.loadData(x_path, "\\s+", true);
		FloatMatrix y = MatrixUtil.loadData(y_path, "\\s+");

		int num_labels = 10;
		FloatMatrix all_theta = FloatMatrix.zeros(X.columns, num_labels);

		MinFunc minFunc = null;
		BaseModel model = null;
		FloatMatrix theta = null;
		for (int i = 1; i <= num_labels; i++) {

			logger.info("模型 {} 初始化...\n", i);
			model = new LogisticRegression(X, y.eq(i));

			logger.info("执行训练...\n");
			minFunc = new LBFGS(model, 50);
			theta = minFunc.train();
			all_theta.putColumn(i - 1, theta);
		}

		logger.info("准确度测算...\n");
		int[] index = X.mmul(all_theta).rowArgmaxs();
		float[] pred = new float[index.length];

		for (int i = 0; i < index.length; i++)
			pred[i] = index[i] + 1f;

		float p = y.eq(new FloatMatrix(pred)).mean() * 100;

		logger.info("训练完成.\n\t all_theta = {} \n\t 准确度 = {}% \n", new Object[] { all_theta, p });

		toc();
	}
}