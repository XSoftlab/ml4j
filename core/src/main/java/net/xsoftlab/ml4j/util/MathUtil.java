package net.xsoftlab.ml4j.util;

import java.util.Random;

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
	 * 计算数字的sigmod
	 * 
	 * @param z
	 *            要计算的数字
	 * @return 计算好的数字
	 */
	public static float sigmoid(float z) {

		return (float) (1f / (1f + Math.exp(-z)));
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

}
