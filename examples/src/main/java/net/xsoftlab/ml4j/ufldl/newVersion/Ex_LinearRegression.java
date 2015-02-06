package net.xsoftlab.ml4j.ufldl.newVersion;

import java.io.IOException;

import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex_LinearRegression extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		logger.info("加载数据...\n");
		String path = System.getProperty("user.dir") + "/resources/ufldl/newVersion/housing.data";
		FloatMatrix matrixs = MatrixUtil.loadData(path, "\\s+", false);

		logger.info("打乱数据...\n");
		matrixs = MatrixUtil.shuffle(matrixs);
		
		/*logger.info("特征标准化...\n");
		FeatureNormalize featureNormalize = new FeatureNormalize(matrixs[0], true);
		FloatMatrix x = featureNormalize.normalize();

		logger.info("执行训练...\n");
		LinearRegression lr = new LinearRegression(x, matrixs[1], 0.3f, 100);
		FloatMatrix theta = lr.train();

		logger.info("计算均方差...\n");
		float rms = MathUtil.std(x.mmul(theta), matrixs[1]);

		logger.info("训练完成.\n\t theta = {} \n\t RMS = {}\n", new Object[] { theta, rms });*/

		toc();
	}
}