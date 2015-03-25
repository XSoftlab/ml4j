package net.xsoftlab.ml4j.coursera.ml.ex3;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import net.xsoftlab.ml4j.minfunc.BFGS;
import net.xsoftlab.ml4j.minfunc.MinFunc;
import net.xsoftlab.ml4j.model.BaseModel;
import net.xsoftlab.ml4j.model.supervised.LogisticRegression;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex3_OneVsAll_MultiThread extends TestUtil {

	public static void main(String[] args) throws IOException, InterruptedException {

		tic();

		logger.info("加载数据...\n");

		String x_path = RESOURCES_PATH + "/coursera/ml/ex3/X.data";
		String y_path = RESOURCES_PATH + "/coursera/ml/ex3/y.data";

		FloatMatrix X = MatrixUtil.loadData(x_path, "\\s+", true);
		FloatMatrix y = MatrixUtil.loadData(y_path, "\\s+");

		int num_labels = 10;
		FloatMatrix all_theta = FloatMatrix.zeros(X.columns, num_labels);

		BaseModel model = null;

		// 线程池
		ExecutorService executor = Executors.newCachedThreadPool();
		CountDownLatch latch = new CountDownLatch(num_labels);// 线程计数

		for (int i = 1; i <= num_labels; i++) {

			logger.info("模型 {} 初始化...\n", i);
			model = new LogisticRegression(X, y.eq(i));

			logger.info("执行训练...\n");
			executor.execute(new OneVsAllThread(latch, all_theta, i, model));
		}

		latch.await();// 等待全部线程执行结束
		executor.shutdown();// 关闭线程池

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

class OneVsAllThread implements Runnable {

	MinFunc minFunc = null;
	BaseModel model = null;
	FloatMatrix theta = null;

	int i;
	FloatMatrix all_theta;

	CountDownLatch latch;

	public OneVsAllThread(CountDownLatch latch, FloatMatrix all_theta, int i, BaseModel model) {
		super();
		this.latch = latch;
		this.all_theta = all_theta;
		this.i = i;
		this.model = model;
	}

	@Override
	public void run() {

		minFunc = new BFGS(model);
		theta = minFunc.train();
		synchronized (all_theta) {
			all_theta.putColumn(i - 1, theta);
		}

		latch.countDown();// 计数器减少1
	}
}