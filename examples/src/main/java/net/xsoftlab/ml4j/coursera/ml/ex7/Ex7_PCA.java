package net.xsoftlab.ml4j.coursera.ml.ex7;

import java.io.IOException;

import net.xsoftlab.ml4j.common.FeatureNormalize;
import net.xsoftlab.ml4j.model.unsupervised.PCA;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex7_PCA extends TestUtil {

	public static void part1() throws IOException {

		logger.info("加载数据...\n");
		String path = COURSE_ML_PATH + "/ex7/ex7data1.data";
		FloatMatrix X = MatrixUtil.loadData(path, "\\s+");

		logger.info("执行特征标准化...\n");
		FeatureNormalize normalize = new FeatureNormalize(X);
		X = normalize.normalize();

		logger.info("初始化pca...\n");
		PCA pca = new PCA(X);

		logger.info("执行pca数据压缩...\n");
		int k = 1;
		FloatMatrix Z = pca.projectData(k);
		logger.info("Z = {}", Z);

		logger.info("执行pca数据恢复...\n");
		FloatMatrix X_rec = pca.recoverData(Z, k);
		logger.info("X_rec = {}", X_rec);
	}

	public static void part2() throws IOException {

		logger.info("加载数据...\n");
		String path = COURSE_ML_PATH + "/ex7/ex7faces.data";
		FloatMatrix X = MatrixUtil.loadData(path, "\\s+");

		logger.info("执行特征标准化...\n");
		FeatureNormalize normalize = new FeatureNormalize(X);
		X = normalize.normalize();

		logger.info("初始化pca...\n");
		PCA pca = new PCA(X);

		logger.info("执行pca数据压缩...\n");
		int k = 100;
		FloatMatrix Z = pca.projectData(k);
		logger.info("Z = {}", Z.getRows(new int[] { 0, 1, 2 }));

		logger.info("执行pca数据恢复...\n");
		FloatMatrix X_rec = pca.recoverData(Z, k);
		logger.info("X_rec = {}", X_rec.getRows(new int[] { 0, 1, 2 }));
	}

	public static void main(String[] args) throws IOException {
		part2();
	}
}
