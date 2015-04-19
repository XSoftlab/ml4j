package net.xsoftlab.ml4j.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import net.xsoftlab.ml4j.exception.Ml4jException;
import net.xsoftlab.ml4j.model.supervised.svm.libsvm.Node;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SVM工具类
 * 
 * @author 王彦超
 *
 */
public class SVMUtil {

	public static Logger logger = LoggerFactory.getLogger(SVMUtil.class);

	/**
	 * 将字符串数组转化为Node数组
	 * 
	 * @param data
	 *            字符串数组
	 * @return double数组
	 */
	private static Node[] convert(String[] data) {

		int length = data.length;
		Node[] result = new Node[length];

		for (int i = 0; i < length; i++) {
			result[i] = new Node(i, Double.parseDouble(data[i]));
		}

		return result;
	}

	/**
	 * 从文件中加载数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * 
	 * @return 数据矩阵
	 * @throws IOException
	 */
	public static Node[][] loadX(InputStream filePath, String split) throws IOException {

		String line;
		String[] data;
		Node[][] matrix;
		int numColumns = -1;
		List<Node[]> list = new ArrayList<Node[]>();

		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(filePath));
			while ((line = reader.readLine()) != null) {

				data = line.trim().split(split);
				if (numColumns < 0)
					numColumns = data.length;
				else if (data.length != numColumns) {
					Ml4jException.logAndThrowException("数据列大小不一致");
				}

				list.add(convert(data));
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		matrix = new Node[list.size()][numColumns];
		for (int i = 0; i < list.size(); i++)
			matrix[i] = list.get(i);

		return matrix;
	}

	/**
	 * 从文件中加载数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * @return 数据矩阵
	 */
	public static Node[][] loadX(String filePath, String split) throws IOException {

		return loadX(new FileInputStream(filePath), split);
	}

	/**
	 * 从文件中加载数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * 
	 * @return 数据矩阵
	 * @throws IOException
	 */
	public static double[] loadY(InputStream filePath) throws IOException {

		String line;
		double[] data;
		List<Double> list = new ArrayList<Double>();

		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(filePath));
			while ((line = reader.readLine()) != null) {
				list.add(Double.parseDouble(line.trim()));
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		data = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
			data[i] = list.get(i);

		return data;
	}

	/**
	 * 从文件中加载数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * @return 数据矩阵
	 */
	public static double[] loadY(String filePath) throws IOException {

		return loadY(new FileInputStream(filePath));
	}
}