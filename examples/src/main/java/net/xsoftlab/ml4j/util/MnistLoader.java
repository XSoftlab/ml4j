package net.xsoftlab.ml4j.util;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import net.xsoftlab.ml4j.common.FeatureNormalize;

import org.jblas.FloatMatrix;

/**
 * 加载MNIST数据
 * 
 * @author 王彦超
 *
 */
public class MnistLoader extends TestUtil {

	public static final String PATH = RESOURCES_PATH + "/ufldl/newVersion/ex1/";

	public static final String TRAIN_IMAGE_PATH = PATH + "train-images-idx3-ubyte";
	public static final String TRAIN_LABEL_PATH = PATH + "train-labels-idx1-ubyte";
	public static final String TEST_IMAGE_PATH = PATH + "t10k-images-idx3-ubyte";
	public static final String TEST_LABEL_PATH = PATH + "t10k-labels-idx1-ubyte";

	public static void main(String[] args) throws IOException {

		tic();
		load(true);
		toc();
	}

	/**
	 * 加载MNIST数据
	 * 
	 * @param binary
	 *            是否只加载0和1
	 * @return 加载好的数据
	 * @throws IOException
	 */
	public static Map<String, FloatMatrix[]> load(boolean binary) throws IOException {

		Map<String, FloatMatrix[]> result = new HashMap<String, FloatMatrix[]>();

		// 加载训练集
		FloatMatrix[] train = new FloatMatrix[2];
		FloatMatrix train_X = MNISTReader.loadMNISTImages(TRAIN_IMAGE_PATH);
		FloatMatrix train_y = MNISTReader.loadMNISTLabel(TRAIN_LABEL_PATH);

		if (binary) {
			// Take only the 0 and 1 digits
			int[] y0 = train_y.eq(0).findIndices();
			int[] y1 = train_y.eq(1).findIndices();
			train_X = MatrixUtil.merge(train_X.getRows(y0), train_X.getRows(y1));
			train_y = MatrixUtil.merge(train_y.getRows(y0), train_y.getRows(y1));
		}

		// Randomly shuffle the data
		int[] rindices = MathUtil.randperm(train_y.length);
		train_X = train_X.getRows(rindices);
		train_y = train_y.getRows(rindices);
		// We standardize the data so that each pixel will have roughly zero
		// mean and unit variance.
		FeatureNormalize normalize = new FeatureNormalize(train_X, true);
		train_X = normalize.normalize();
		train[0] = train_X;
		train[1] = train_y;

		result.put("train", train);

		// 加载测试集
		FloatMatrix[] test = new FloatMatrix[2];
		FloatMatrix test_X = MNISTReader.loadMNISTImages(TEST_IMAGE_PATH);
		FloatMatrix test_y = MNISTReader.loadMNISTLabel(TEST_LABEL_PATH);

		if (binary) {
			// Take only the 0 and 1 digits
			int[] y0 = test_y.eq(0).findIndices();
			int[] y1 = test_y.eq(1).findIndices();
			test_X = MatrixUtil.merge(test_X.getRows(y0), test_X.getRows(y1));
			test_y = MatrixUtil.merge(test_y.getRows(y0), test_y.getRows(y1));
		}

		// Randomly shuffle the data
		rindices = MathUtil.randperm(test_y.length);
		test_X = test_X.getRows(rindices);
		test_y = test_y.getRows(rindices);
		// Standardize using the same mean and scale as the training data.
		normalize = new FeatureNormalize(test_X, normalize.getMu(), normalize.getSigma(), true);
		test_X = normalize.normalize();
		test[0] = test_X;
		test[1] = test_y;

		result.put("test", test);

		return result;
	}
}
