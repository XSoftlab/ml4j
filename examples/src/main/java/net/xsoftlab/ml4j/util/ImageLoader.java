package net.xsoftlab.ml4j.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.jblas.FloatMatrix;

/**
 * 加载MNIST数据
 * 
 * @author 王彦超
 *
 */
public class ImageLoader extends TestUtil {

	public static void main(String[] args) throws IOException {

		tic();

		String path = RESOURCES_PATH + "/coursera/ml/ex7/bird_small.png";
		load(path, false);

		toc();
	}

	/**
	 * 加载图片至FloatMatrix - 不除以255
	 * 
	 * @param path 图片路径
	 * @return FloatMatrix(rows,3)
	 *         FloatMatrix[:,0] R通道颜色
	 *         FloatMatrix[:,1] G通道颜色
	 *         FloatMatrix[:,2] B通道颜色
	 * @throws IOException
	 */
	public static MatrixImage load(String path) throws IOException {
		return load(path, false);
	}

	/**
	 * 加载图片至FloatMatrix
	 * 
	 * @param path 图片路径
	 * @return FloatMatrix(rows,3)
	 *         FloatMatrix[:,0] R通道颜色
	 *         FloatMatrix[:,1] G通道颜色
	 *         FloatMatrix[:,2] B通道颜色
	 * @throws IOException
	 */
	public static MatrixImage load(String path, boolean divide) throws IOException {

		FloatMatrix result = null;

		BufferedImage bi = ImageIO.read(new File(path));
		int width = bi.getWidth();
		int height = bi.getHeight();
		int size = width * height;

		int[] buffer = new int[size];
		bi.getRGB(0, 0, bi.getWidth(), bi.getHeight(), buffer, 0, height);
		// ImShow.show(buffer, width, height, 1);

		int gt, bt;
		float r[] = new float[size];
		float g[] = new float[size];
		float b[] = new float[size];
		float[] data = new float[size * 3];

		for (int i = 0; i < size; i++) {
			r[i] = 0xFF & buffer[i];
			gt = 0xFF00 & buffer[i];
			g[i] = gt >> 8;
			bt = 0xFF0000 & buffer[i];
			b[i] = bt >> 16;
		}
		System.arraycopy(r, 0, data, 0, size);
		System.arraycopy(g, 0, data, size, size);
		System.arraycopy(b, 0, data, size * 2, size);

		result = new FloatMatrix(size, 3, data);

		// ImShow.show(result, width, height, false);
		return new MatrixImage(width, height, divide ? result.div(255) : result);
	}

	/**
	 * 矩阵图像
	 * 
	 * @author X
	 *
	 * @data 2015年4月22日
	 */
	public static class MatrixImage {

		private int width;
		private int height;
		private FloatMatrix matrix;

		public MatrixImage(int width, int height, FloatMatrix matrix) {
			super();
			this.width = width;
			this.height = height;
			this.matrix = matrix;
		}

		public int getWidth() {
			return width;
		}

		public void setWidth(int width) {
			this.width = width;
		}

		public int getHeight() {
			return height;
		}

		public void setHeight(int height) {
			this.height = height;
		}

		public FloatMatrix getMatrix() {
			return matrix;
		}

		public void setMatrix(FloatMatrix matrix) {
			this.matrix = matrix;
		}
	}
}
