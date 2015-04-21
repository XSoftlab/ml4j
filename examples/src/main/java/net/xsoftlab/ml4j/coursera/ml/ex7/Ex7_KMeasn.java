package net.xsoftlab.ml4j.coursera.ml.ex7;

import java.io.IOException;

import net.xsoftlab.ml4j.model.unsupervised.KMeans;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex7_KMeasn extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = RESOURCES_PATH + "/coursera/ml/ex7/kmeans.data";
		FloatMatrix X = MatrixUtil.loadData(path, "\\s+");

		logger.info("模型初始化...\n");
		KMeans kMeans = new KMeans(X, 3);

		logger.info("执行训练...\n");
		FloatMatrix centroids = kMeans.run(10);

		logger.info("运行完毕.\n聚类中心如下：\n{}\n", centroids);
		toc();
	}
}
