package net.xsoftlab.ml4j.coursera.ml.ex6;

import java.io.IOException;

import net.xsoftlab.ml4j.model.supervised.svm.libsvm.Model;
import net.xsoftlab.ml4j.model.supervised.svm.libsvm.Node;
import net.xsoftlab.ml4j.model.supervised.svm.libsvm.Parameter;
import net.xsoftlab.ml4j.model.supervised.svm.libsvm.Problem;
import net.xsoftlab.ml4j.model.supervised.svm.libsvm.SVM;
import net.xsoftlab.ml4j.util.SVMUtil;
import net.xsoftlab.ml4j.util.TestUtil;

public class Ex6_SVM extends TestUtil {

	public static Model train(Problem problem) {

		Parameter param = new Parameter();

		// default values
		param.svm_type = Parameter.C_SVC;
		param.kernel_type = Parameter.RBF;
		param.degree = 3;
		param.gamma = 0;
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 40;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];

		param.kernel_type = Parameter.LINEAR;
		param.C = 0.1;

		return SVM.svm_train(problem, param);
	}

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载训练数据...\n");
		String x_path = RESOURCES_PATH + "/coursera/ml/ex6/X.data";
		String y_path = RESOURCES_PATH + "/coursera/ml/ex6/y.data";
		Node[][] X = SVMUtil.loadX(x_path, "\\s+");
		double[] y = SVMUtil.loadY(y_path);

		logger.info("训练中...\n");
		Problem problem = new Problem(X, y);
		Model model = train(problem);

		logger.info("加载测试数据...\n");
		String test_x_path = RESOURCES_PATH + "/coursera/ml/ex6/Xtest.data";
		String test_y_path = RESOURCES_PATH + "/coursera/ml/ex6/ytest.data";
		Node[][] test_X = SVMUtil.loadX(test_x_path, "\\s+");
		double[] test_y = SVMUtil.loadY(test_y_path);

		logger.info("测试中...\n");
		int trueCount = 0;
		int falseCount = 0;
		double result = 0;
		int length = test_y.length;
		for (int i = 0; i < length; i++) {
			result = SVM.svm_predict(model, test_X[i]);
			if (result == test_y[i])
				trueCount++;
			else
				falseCount++;
		}
		logger.info("共有 {} 条测试数据,正确 {} 条,错误 {} 条,正确率 {}%", new Object[] { length, trueCount, falseCount,
				(float) trueCount / (float) length * 100 });
	}
}
