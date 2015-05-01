package net.xsoftlab.ml4j.coursera.ml.ex7;

import java.io.IOException;

import net.xsoftlab.ml4j.model.unsupervised.KMeans;
import net.xsoftlab.ml4j.util.ImShow;
import net.xsoftlab.ml4j.util.ImageLoader;
import net.xsoftlab.ml4j.util.ImageLoader.MatrixImage;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex7_KMeasn extends TestUtil {

	/**
	 * 迭代并找出聚类中心
	 * 
	 * @throws IOException
	 */
	public static void part1() throws IOException {
		tic();

		logger.info("加载数据...\n");
		String path = COURSE_ML_PATH + "/ex7/ex7data2.data";
		FloatMatrix X = MatrixUtil.loadData(path, "\\s+");

		logger.info("模型初始化...\n");
		KMeans kMeans = new KMeans(X, 3);

		logger.info("执行训练...\n");
		FloatMatrix centroids = kMeans.run(10);

		logger.info("运行完毕.\n聚类中心如下：\n{}\n", centroids);
		toc();
	}

	/**
	 * 图象压缩
	 * 
	 * @throws IOException
	 */
	public static void part2() throws IOException {
		tic();

		logger.info("加载数据...\n");
		String path = COURSE_ML_PATH + "/ex7/bird_small.png";
		MatrixImage mi = ImageLoader.load(path);
		FloatMatrix X = mi.getMatrix();

		logger.info("模型初始化...\n");
		KMeans kMeans = new KMeans(X, 16);

		logger.info("执行训练...\n");
		FloatMatrix centroids = kMeans.run(10);

		logger.info("运行完毕.\n聚类中心如下：\n{}\n", centroids);

		logger.info("图象对比...\n");
		FloatMatrix indices = kMeans.findClosestCentroids();
		int[] index = indices.toIntArray();
		FloatMatrix result = centroids.getRows(index);

		ImShow.show(mi);

		mi.setMatrix(result);
		ImShow.show(mi);
		toc();
	}

	public static void main(String[] args) throws IOException {
		part2();
	}
}
