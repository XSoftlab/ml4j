package net.xsoftlab.ml4j.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.xsoftlab.ml4j.exception.Ml4jException;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 矩阵工具 - jblas拓展
 * 
 * @author 王彦超
 *
 */
public class MatrixUtil {

	public static Logger logger = LoggerFactory.getLogger(MatrixUtil.class);

	/**
	 * 将数据保存到文件
	 * 
	 * @param fileName
	 *            文件路径
	 */
	public static void saveData(FloatMatrix matrix, String fileName) throws IOException {

		File file = new File(fileName); // 创建文件
		Writer writer = null;
		try {
			writer = new OutputStreamWriter(new FileOutputStream(file)); // 打开文件输出流
			writer.write(matrix.toString()); // 写入文件
		} finally {
			if (writer != null)
				writer.close();
		}
	}

	/**
	 * 将字符串数组转化为float数组
	 * 
	 * @param data
	 *            字符串数组
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return float数组
	 */
	private static float[] convert(String[] data, boolean intercept) {

		int length = intercept ? data.length + 1 : data.length;
		float[] result = new float[length];

		for (int i = 0; i < length; i++) {
			if (intercept) {
				if (i == 0) {
					result[i] = 1;
				} else {
					result[i] = Float.parseFloat(data[i - 1]);
				}
			} else
				result[i] = Float.parseFloat(data[i]);
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
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return 数据矩阵
	 * @throws IOException
	 */
	public static FloatMatrix loadData(InputStream filePath, String split, boolean intercept) throws IOException {

		String line;
		String[] data;
		FloatMatrix matrix;
		int numColumns = -1;
		List<float[]> list = new ArrayList<float[]>();

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

				list.add(convert(data, intercept));
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		matrix = new FloatMatrix(list.size(), numColumns);
		for (int i = 0; i < list.size(); i++)
			matrix.putRow(i, new FloatMatrix(list.get(i)));

		return matrix;
	}

	/**
	 * 从文件中加载数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return 数据矩阵
	 */
	public static FloatMatrix loadData(String filePath, String split, boolean intercept) throws IOException {

		return loadData(new FileInputStream(filePath), split, intercept);
	}

	/**
	 * 从文件中加载数据   - 添加截距项
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * 
	 * @return 数据矩阵
	 */
	public static FloatMatrix loadData(String filePath, String split) throws IOException {

		return loadData(new FileInputStream(filePath), split, true);
	}

	/**
	 * 将字符串数组转化为float数组
	 * 
	 * @param data
	 *            字符串数组
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return float数组集合
	 */
	private static List<float[]> convertWithXY(String[] data, boolean intercept) {

		List<float[]> list = new ArrayList<float[]>();
		int length = intercept ? data.length + 1 : data.length;

		float[] result0 = new float[length - 1];
		float[] result1 = new float[1];

		for (int i = 0; i < length; i++) {
			if (intercept) {
				if (i == 0)
					result0[i] = 1;
				else if (i != length - 1)
					result0[i] = Float.parseFloat(data[i - 1]);
				else
					result1[0] = Float.parseFloat(data[i - 1]);
			} else {
				if (i != length - 1)
					result0[i] = Float.parseFloat(data[i]);
				else
					result1[0] = Float.parseFloat(data[i]);
			}
		}

		list.add(result0);
		list.add(result1);

		return list;
	}

	/**
	 * 从文件中加载包含X、Y的数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return 数据矩阵数组
	 * @throws IOException
	 */
	public static FloatMatrix[] loadDataWithXY(InputStream filePath, String split, boolean intercept)
			throws IOException {

		String line;
		String[] data;
		FloatMatrix[] matrixs = new FloatMatrix[2];
		int numColumns = -1;
		List<float[]> list;
		List<float[]> resList;
		Map<String, List<float[]>> map = new HashMap<String, List<float[]>>();

		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(filePath));
			while ((line = reader.readLine()) != null) {

				data = line.trim().split(split);
				if (numColumns < 0)
					numColumns = data.length;
				else if (data.length != numColumns) {
					Ml4jException.logAndThrowException("数据列大小不一致！");
				}

				resList = convertWithXY(data, intercept);
				if (map.containsKey("0")) {
					list = map.get("0");
					list.add(resList.get(0));
					map.put("0", list);
				} else {
					list = new ArrayList<float[]>();
					list.add(resList.get(0));
					map.put("0", list);
				}

				if (map.containsKey("1")) {
					list = map.get("1");
					list.add(resList.get(1));
					map.put("1", list);
				} else {
					list = new ArrayList<float[]>();
					list.add(resList.get(1));
					map.put("1", list);
				}
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		numColumns = intercept ? numColumns : numColumns - 1;

		List<float[]> map0 = map.get("0");
		matrixs[0] = new FloatMatrix(map0.size(), numColumns);
		for (int i = 0; i < map0.size(); i++)
			matrixs[0].putRow(i, new FloatMatrix(map0.get(i)));

		List<float[]> map1 = map.get("1");
		matrixs[1] = new FloatMatrix(map1.size(), 1);
		for (int i = 0; i < map1.size(); i++)
			matrixs[1].putRow(i, new FloatMatrix(map1.get(i)));

		return matrixs;
	}

	/**
	 * 从文件中加载包含X、Y的数据
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * @param intercept
	 *            是否添加截距项
	 * 
	 * @return 数据矩阵数组
	 */
	public static FloatMatrix[] loadDataWithXY(String filePath, String split, boolean intercept) throws IOException {

		return loadDataWithXY(new FileInputStream(filePath), split, intercept);
	}

	/**
	 * 从文件中加载包含X、Y的数据 - 添加截距项
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * 
	 * @return 数据矩阵数组
	 */
	public static FloatMatrix[] loadDataWithXY(String filePath, String split) throws IOException {

		return loadDataWithXY(new FileInputStream(filePath), split, true);
	}

	/**
	 * 添加截距项
	 * 
	 * @param matrix
	 *            要添加截距项的矩阵
	 * @return 添加过截距项的矩阵
	 */
	public static FloatMatrix addIntercept(FloatMatrix matrix) {

		return add(FloatMatrix.ones(matrix.rows), matrix, 2);
	}

	/**
	 * 矩阵合并 (按行)
	 * 
	 * @param matrix
	 *            原始矩阵
	 * @param additional
	 *            要合并的矩阵
	 * @return 合并后的矩阵
	 */
	public static FloatMatrix add(FloatMatrix matrix, FloatMatrix additional) {

		return add(matrix, additional, 1);
	}

	/**
	 * 矩阵合并
	 * 
	 * @param matrix
	 *            原始矩阵
	 * @param additional
	 *            要合并的矩阵
	 * @param dim
	 *            1/按行合并 2/按列合并
	 * @return 合并后的矩阵
	 */
	public static FloatMatrix add(FloatMatrix matrix, FloatMatrix additional, int dim) {

		int rows = matrix.rows;
		int columns = matrix.columns;

		int aRows = additional.rows;
		int aColumns = additional.columns;

		FloatMatrix result = null;

		if (dim == 1) {
			if (columns != aColumns)
				Ml4jException.logAndThrowException("要添加的矩阵与原矩阵列数不符。");

			result = new FloatMatrix(rows + aRows, columns);
			for (int i = 0; i < rows; i++) {
				result.putRow(i, matrix.getRow(i));
			}
			for (int i = 0; i < aRows; i++) {
				result.putRow(i + rows, additional.getRow(i));
			}
		} else if (dim == 2) {
			if (rows != aRows)
				Ml4jException.logAndThrowException("要添加的矩阵与原矩阵行数不符。");

			result = new FloatMatrix(rows, columns + aColumns);
			for (int i = 0; i < columns; i++) {
				result.putColumn(i, matrix.getColumn(i));
			}
			for (int i = 0; i < aColumns; i++) {
				result.putColumn(i + columns, additional.getColumn(i));
			}
		}

		return result;
	}

	/**
	 * 打乱矩阵 - 按行打乱
	 * 
	 * @param matrix
	 *            要打乱的矩阵
	 * @return 打乱后的矩阵
	 */
	public static FloatMatrix shuffle(FloatMatrix matrix) {

		int[] rindices = MatrixUtil.randperm(matrix.rows);
		return matrix.getRows(rindices);
	}

	/**
	 * 随机打乱一组数字
	 * 
	 * @param number
	 *            要打乱的数字范围（从0开始）
	 * @return 打乱好的数组
	 */
	public static int[] randperm(int number) {

		int[] array = new int[number];
		for (int i = 0; i < number; i++)
			array[i] = i;

		int index, temp;
		Random random = new Random();
		for (int i = number; i > 1; i--) {
			index = random.nextInt(i);
			temp = array[i - 1];
			array[i - 1] = array[index];
			array[index] = temp;
		}

		return array;
	}

	/**
	 * 计算矩阵的sigmoid
	 * 
	 * @param matrix
	 *            要计算的矩阵
	 * @return 计算好的矩阵
	 */
	public static FloatMatrix sigmoid(FloatMatrix matrix) {

		int rows = matrix.rows;
		int columns = matrix.columns;
		FloatMatrix result = new FloatMatrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				result.put(i, j, MathUtil.sigmoid(matrix.get(i, j)));
			}
		}

		return result;
	}

	/**
	 * 计算矩阵的log
	 * 
	 * @param matrix
	 *            要计算的矩阵
	 * @return 计算好的矩阵
	 */
	public static FloatMatrix log(FloatMatrix matrix) {

		int rows = matrix.rows;
		int columns = matrix.columns;
		Double log = 0d;
		FloatMatrix result = new FloatMatrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				log = Math.log(matrix.get(i, j));
				if (log.isInfinite() && log < 0)
					result.put(i, j, 0f);
				else
					result.put(i, j, log.floatValue());
			}
		}

		return result;
	}

	/**
	 * 计算向量的开方
	 * 
	 * @param vector
	 *            要计算的向量
	 * @return 计算好的向量
	 */
	public static FloatMatrix pow(FloatMatrix vector, int time) {

		if (!vector.isVector())
			Ml4jException.logAndThrowException("参数vector必须是向量！");
		else if (time < 0) {
			Ml4jException.logAndThrowException("参数time必须是大于或等于0的整数！");
		}

		if (time == 0)
			return FloatMatrix.ones(vector.rows, vector.columns);

		FloatMatrix result = vector.dup();// copy矩阵
		for (int i = 1; i < time; i++) {
			result.muli(vector);
		}

		return result;
	}
}