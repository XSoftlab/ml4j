package net.xsoftlab.ml4j.coursera.ml.ex8;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import net.xsoftlab.ml4j.model.supervised.BaseModel;
import net.xsoftlab.ml4j.model.supervised.CollaborativeFiltering;
import net.xsoftlab.ml4j.util.MatrixUtil;
import net.xsoftlab.ml4j.util.TestUtil;

import org.jblas.FloatMatrix;

public class Ex8_RecommenderSystems extends TestUtil {

	public static void part1() throws IOException {

		logger.info("加载数据...\n");
		String path_X = COURSE_ML_PATH + "/ex8/ex8_movies_X.data";
		String path_Theta = COURSE_ML_PATH + "/ex8/ex8_movies_Theta.data";
		String path_Y = COURSE_ML_PATH + "/ex8/ex8_movies_Y.data";

		FloatMatrix X = MatrixUtil.loadData(path_X, "\\s+");
		FloatMatrix theta = MatrixUtil.loadData(path_Theta, "\\s+");
		FloatMatrix y = MatrixUtil.loadData(path_Y, "\\s+");
		FloatMatrix r = y.gt(0);

		int num_users = 4;
		int num_movies = 5;
		int num_features = 3;
		X = X.getRange(0, num_movies, 0, num_features);
		theta = theta.getRange(0, num_users, 0, num_features);
		y = y.getRange(0, num_movies, 0, num_users);
		r = r.getRange(0, num_movies, 0, num_users);

		// Evaluate cost function
		CollaborativeFiltering cf = new CollaborativeFiltering(y, r, num_features);
		float J = (float) cf.compute(MatrixUtil.merge(X.data, theta.data), 1);

		logger.info("Cost at loaded parameters: {} \n(this value should be about 22.22)\n", J);

		CollaborativeFiltering cf1 = new CollaborativeFiltering(y, r, num_features, 1.5f);
		J = (float) cf1.compute(MatrixUtil.merge(X.data, theta.data), 3);

		logger.info("Cost at loaded parameters(lambda = 1.5): {} \n(this value should be about 31.34)\n", J);
	}

	public static void part2() throws IOException {

		logger.info("加载数据...\n");
		String path_Y = COURSE_ML_PATH + "/ex8/ex8_movies_Y.data";
		String path_ids = COURSE_ML_PATH + "/ex8/movie_ids.txt";

		logger.info("添加你对的电影的评分...\n");
		FloatMatrix myRatings = getMyRatings();
		FloatMatrix y = MatrixUtil.loadData(path_Y, "\\s+");
		y = MatrixUtil.merge(myRatings, y, 2);
		FloatMatrix r = y.gt(0);

		logger.info("模型初始化...\n");
		int numFeatures = 10;
		BaseModel model = new CollaborativeFiltering(y, r, numFeatures, 10);

		logger.info("Normalize Ratings...\n");
		FloatMatrix yMean = ((CollaborativeFiltering) model).normalizeRatings();

		logger.info("执行训练...\n");
		FloatMatrix params = model.train();

		logger.info("训练完毕...\n");
		int numUsers = y.columns;
		int numMovies = y.rows;

		FloatMatrix x = params.getRange(0, numMovies * numFeatures);
		FloatMatrix theta = params.getRange(numMovies * numFeatures, params.length);

		x = x.reshape(numMovies, numFeatures);
		theta = theta.reshape(numUsers, numFeatures);

		logger.info("加载电影列表...\n");
		Map<Integer, String> movieList = loadMovieList(path_ids);

		logger.info("为你推荐的电影...\n");
		FloatMatrix p = x.mmul(theta.transpose());
		FloatMatrix myPredictions = p.getColumn(0).add(yMean);

		int[] indexes = myPredictions.sortingPermutation();
		int length = indexes.length;

		logger.info("Top recommendations for you:\n");
		for (int i = 1; i <= 10; i++) {
			int j = indexes[length - i];
			logger.info("Predicting rating {} for movie {}\n", new Object[] { myPredictions.get(j), movieList.get(j) });
		}

		logger.info("Original ratings provided:\n");
		for (int i = 0; i < myRatings.length; i++) {
			if (myRatings.get(i) > 0) {
				logger.info("Rated {} for {}\n", new Object[] { myRatings.get(i), movieList.get(i) });
			}
		}
	}

	public static FloatMatrix getMyRatings() {

		// Initialize my ratings
		FloatMatrix myRatings = FloatMatrix.zeros(1682, 1);

		// We have selected a few movies we liked / did not like and the ratings
		// we gave are as follows:
		myRatings.put(0, 4);
		myRatings.put(97, 2);
		myRatings.put(6, 3);
		myRatings.put(11, 5);
		myRatings.put(53, 4);
		myRatings.put(63, 5);
		myRatings.put(65, 3);
		myRatings.put(68, 5);
		myRatings.put(182, 4);
		myRatings.put(225, 5);
		myRatings.put(354, 5);

		return myRatings;
	}

	public static Map<Integer, String> loadMovieList(String filePath) throws IOException {

		int i = 0;
		String line;
		BufferedReader reader = null;
		Map<Integer, String> result = new HashMap<Integer, String>();

		try {
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
			while ((line = reader.readLine()) != null) {
				result.put(i++, line.replaceAll("^\\d+\\s", ""));
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		return result;
	}

	public static void main(String[] args) throws IOException {
		part2();
	}
}
