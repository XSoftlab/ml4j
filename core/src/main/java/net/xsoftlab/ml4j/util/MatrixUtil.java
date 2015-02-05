package net.xsoftlab.ml4j.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
	 */
	public static FloatMatrix loadData(InputStream filePath, String split, boolean intercept) throws IOException {

		String line;
		String[] data;
		FloatMatrix matrix;
		int numColumns = -1;
		List<float[]> list = new ArrayList<float[]>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(filePath));

		while ((line = reader.readLine()) != null) {

			data = line.trim().split(split);
			if (numColumns < 0)
				numColumns = data.length;
			else if (data.length != numColumns) {
				logger.error("数据前后大小不一致！");
				throw new Ml4jException("数据前后大小不一致！");
			}

			list.add(convert(data, intercept));
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
		BufferedReader reader = new BufferedReader(new InputStreamReader(filePath));

		while ((line = reader.readLine()) != null) {

			data = line.trim().split(split);
			if (numColumns < 0)
				numColumns = data.length;
			else if (data.length != numColumns) {
				logger.error("数据前后大小不一致！");
				throw new Ml4jException("数据前后大小不一致！");
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
	 * 添加截距项
	 * 
	 * @param x
	 *            要添加截距项的矩阵
	 * @return 添加过截距项的矩阵
	 */
	public static FloatMatrix addIntercept(FloatMatrix x) {

		FloatMatrix matrix = new FloatMatrix(x.rows, x.columns + 1);
		matrix.putColumn(0, FloatMatrix.ones(x.rows));
		for (int i = 0; i < x.columns; i++) {
			matrix.putColumn(i + 1, x.getColumn(i));
		}

		return matrix;
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
		float sum;// 和
		float std;
		FloatMatrix mu;// 平均值
		FloatMatrix temp;// 临时变量
		FloatMatrix matrix;// 计算结果

		if (dim == 1) {
			size = x.columns;
			mu = x.columnMeans();
			matrix = new FloatMatrix(1, size);
			for (int i = 0; i < size; i++) {
				temp = x.getColumn(i).sub(mu.get(i));
				sum = temp.transpose().mmul(temp).get(0);
				std = (float) Math.sqrt(sum / (temp.rows - 1));
				matrix.put(i, std);
			}
		} else {
			size = x.rows;
			mu = x.rowMeans();
			matrix = new FloatMatrix(size, 1);
		}

		return matrix;
	}
}
