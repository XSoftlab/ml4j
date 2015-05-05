package net.xsoftlab.ml4j.util;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.xsoftlab.ml4j.exception.Ml4jException;

import org.jblas.Decompose;
import org.jblas.Decompose.LUDecomposition;
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
	 * 将数据保存到文件 - 由于原FloatMartix 的save方法有bug,暂时在这自己封装一个
	 * 
	 * @param fileName
	 *            文件路径
	 */
	public static void save(FloatMatrix matrix, String fileName) throws IOException {

		DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName, false));
		dos.writeUTF("float");
		dos.writeInt(matrix.columns);
		dos.writeInt(matrix.rows);

		float[] data = matrix.data;

		dos.writeInt(data.length);
		for (int i = 0; i < data.length; i++) {
			dos.writeFloat(data[i]);
		}

		dos.close();
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
					Ml4jException.logAndThrowException("数据列大小不一致data.length = " + data.length + ",numColumns = "
							+ numColumns + ",line = " + line + "");
				}

				list.add(convert(data, intercept));
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		numColumns = intercept ? numColumns + 1 : numColumns;

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
	 * 从文件中加载数据 - 不添加截距项
	 * 
	 * @param filePath
	 *            文件路径
	 * @param split
	 *            分隔符
	 * 
	 * @return 数据矩阵
	 */
	public static FloatMatrix loadData(String filePath, String split) throws IOException {

		return loadData(new FileInputStream(filePath), split, false);
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

		return merge(FloatMatrix.ones(matrix.rows), matrix, 2);
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
	public static FloatMatrix merge(FloatMatrix matrix, FloatMatrix additional) {

		return merge(matrix, additional, 1);
	}

	/**
	 * 矩阵合并
	 * 
	 * @param matrix
	 *            原始矩阵数组
	 * @param additional
	 *            要合并的矩阵数组
	 * @return 合并后的矩阵
	 */
	public static FloatMatrix merge(float[] matrix, float[] additional) {

		float[] result = new float[matrix.length + additional.length];
		System.arraycopy(matrix, 0, result, 0, matrix.length);
		System.arraycopy(additional, 0, result, matrix.length, additional.length);

		return new FloatMatrix(result);
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
	public static FloatMatrix merge(FloatMatrix matrix, FloatMatrix additional, int dim) {

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

		int[] rindices = MathUtil.randperm(matrix.rows);
		return matrix.getRows(rindices);
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
	 * 计算矩阵方差(N-1)
	 * 
	 * @param x
	 *            要计算的矩阵
	 * @param dim
	 *            1 计算列标准差，2计算行标准差
	 * @return 计算好的方差矩阵
	 */
	public static FloatMatrix var(FloatMatrix x, int dim) {

		return var(x, true, dim);
	}

	/**
	 * 计算矩阵方差
	 * 
	 * @param x
	 *            要计算的矩阵
	 * @param flag
	 *            true计算除以N-1,false计算除以N
	 * @param dim
	 *            1 计算列标准差，2计算行标准差
	 * @return 计算好的方差矩阵
	 */
	public static FloatMatrix var(FloatMatrix x, boolean flag, int dim) {

		int size;
		FloatMatrix mu;// 平均值
		FloatMatrix std = null;// 计算结果

		if (dim == 1) {
			size = x.columns;
			mu = x.columnMeans();
			std = new FloatMatrix(1, size);
			for (int i = 0; i < size; i++) {
				std.put(i, var(x.getColumn(i), mu.get(new int[] { i }), flag));
			}
		} else if (dim == 2) {
			size = x.rows;
			mu = x.rowMeans();
			std = new FloatMatrix(size, 1);
			for (int i = 0; i < size; i++) {
				std.put(i, var(x.getRow(i), mu.get(new int[] { i }), flag));
			}
		}

		return std;
	}

	/**
	 * 计算两个向量的方差
	 * 
	 * @param vector1
	 *            向量1
	 * @param vector2
	 *            向量2
	 * @param flag
	 *            true计算除以N-1,false计算除以N
	 * @return 方差
	 */
	public static float var(FloatMatrix vector1, FloatMatrix vector2, boolean flag) {

		if (!vector1.isVector() || !vector2.isVector()) {
			Ml4jException.logAndThrowException("参数必须是向量！");
		}

		int n;// 数量，被除数
		float std;// 标准差
		float sum;// 和
		FloatMatrix temp;

		n = flag ? vector1.length - 1 : vector1.length;
		temp = vector1.sub(vector2);
		sum = temp.transpose().mmul(temp).get(0);
		std = (float) sum / (float) n;

		return std;
	}

	/**
	 * 计算矩阵标准差(N-1)
	 * 
	 * @param x
	 *            要计算的矩阵
	 * @param dim
	 *            1 计算列标准差，2计算行标准差
	 * @return 计算好的标准差矩阵
	 */
	public static FloatMatrix std(FloatMatrix x, int dim) {

		return std(x, true, dim);
	}

	/**
	 * 计算矩阵标准差
	 * 
	 * @param x
	 *            要计算的矩阵
	 * @param flag
	 *            true计算除以N-1,false计算除以N
	 * @param dim
	 *            1 计算列标准差，2计算行标准差
	 * @return 计算好的标准差矩阵
	 */
	public static FloatMatrix std(FloatMatrix x, boolean flag, int dim) {

		int size;
		FloatMatrix mu;// 平均值
		FloatMatrix std = null;// 计算结果

		if (dim == 1) {
			size = x.columns;
			mu = x.columnMeans();
			std = new FloatMatrix(1, size);
			for (int i = 0; i < size; i++) {
				std.put(i, std(x.getColumn(i), mu.get(new int[] { i }), flag));
			}
		} else if (dim == 2) {
			size = x.rows;
			mu = x.rowMeans();
			std = new FloatMatrix(size, 1);
			for (int i = 0; i < size; i++) {
				std.put(i, std(x.getRow(i), mu.get(new int[] { i }), flag));
			}
		}

		return std;
	}

	/**
	 * 计算两个向量的标准差
	 * 
	 * @param vector1
	 *            向量1
	 * @param vector2
	 *            向量2
	 * @param flag
	 *            true计算除以N-1,false计算除以N
	 * @return 标准差
	 */
	public static float std(FloatMatrix vector1, FloatMatrix vector2, boolean flag) {

		if (!vector1.isVector() || !vector2.isVector()) {
			Ml4jException.logAndThrowException("参数必须是向量！");
		}

		int n;// 数量，被除数
		float std;// 标准差
		float sum;// 和
		FloatMatrix temp;

		n = flag ? vector1.length - 1 : vector1.length;
		temp = vector1.sub(vector2);
		sum = temp.transpose().mmul(temp).get(0);
		std = (float) Math.sqrt(sum / n);

		return std;
	}

	/**
	 * 计算两个向量的标准差(N-1)
	 * 
	 * @param vector1
	 *            向量1
	 * @param vector2
	 *            向量2
	 * @return 标准差
	 */
	public static float std(FloatMatrix vector1, FloatMatrix vector2) {

		return std(vector1, vector2, true);
	}

	/**
	 * 计算矩阵行列式
	 * 
	 * @param matrix 要计算的矩阵
	 * @return 行列式结果
	 */
	public static float det(FloatMatrix matrix) {

		LUDecomposition<FloatMatrix> lup = Decompose.lu(matrix);
		return lup.u.diag().prod();
	}
}