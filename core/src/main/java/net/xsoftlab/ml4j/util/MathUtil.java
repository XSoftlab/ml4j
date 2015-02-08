package net.xsoftlab.ml4j.util;

import net.xsoftlab.ml4j.exception.Ml4jException;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 数学工具
 * 
 * @author 王彦超
 *
 */
public class MathUtil {

	public static Logger logger = LoggerFactory.getLogger(MathUtil.class);

	/**
	 * 计算矩阵标准差(又称均方差)(N-1)
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
	 * 计算矩阵标准差(又称均方差)
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
		FloatMatrix std;// 计算结果

		if (dim == 1) {
			size = x.columns;
			mu = x.columnMeans();
			std = new FloatMatrix(1, size);
			for (int i = 0; i < size; i++) {
				std.put(i, std(x.getColumn(i), mu.get(new int[] { i }), flag));
			}
		} else {
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
	 * 计算两个向量的标准差(又称均方差)
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
	 * 计算两个向量的标准差(又称均方差)(N-1)
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
	 * 计算数字的sigmod
	 * 
	 * @param z
	 *            要计算的数字
	 * @return 计算好的数字
	 */
	public static float sigmoid(float z) {

		return (float) (1 / (1 + Math.exp(-z)));
	}
}
