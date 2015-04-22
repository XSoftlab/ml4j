package net.xsoftlab.ml4j.util;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;

import net.xsoftlab.ml4j.util.ImageLoader.MatrixImage;

import org.jblas.FloatMatrix;

public class ImShow extends JPanel {

	private static final long serialVersionUID = 1107409750852921744L;

	private static final int TYPE = BufferedImage.TYPE_INT_RGB;

	private BufferedImage image;

	public ImShow(BufferedImage image) {
		this.image = image;
	}

	public static void show(int[] buffer, int width, int height, int count) {

		BufferedImage bufferedImage = new BufferedImage(width, height * count, TYPE);
		bufferedImage.setRGB(0, 0, width, height * count, buffer, 0, width);
		JFrame jframe = new JFrame();
		jframe.add(new ImShow(bufferedImage));
		jframe.setSize(200, 200);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setVisible(true);
	}

	public static void show(float[] buffer, int width, int height, int count) {

		int[] iBuffer = new int[buffer.length];
		for (int i = 0; i < buffer.length; i++)
			iBuffer[i] = (int) buffer[i];

		show(iBuffer, width, height, count);
	}

	/**
	 * 显示RGB矩阵图像 - 不乘以255
	 * 
	 * @param mi 矩阵图像
	 */
	public static void show(MatrixImage mi) {
		show(mi, false);
	}

	/**
	 * 显示RGB矩阵图像
	 * 
	 * @param mi 矩阵图像
	 * @param mul 是否需要乘以255
	 */
	public static void show(MatrixImage mi, boolean mul) {

		int width = mi.getWidth();
		int height = mi.getHeight();
		FloatMatrix matrix = mi.getMatrix();

		int size = matrix.rows;// width * height
		matrix = mul ? matrix.mul(255) : matrix;

		int rt, gt, bt;
		float r[] = matrix.getColumn(0).data;
		float g[] = matrix.getColumn(1).data;
		float b[] = matrix.getColumn(2).data;

		int[] buffer = new int[size];
		for (int i = 0; i < size; i++) {
			bt = (int) b[i];
			gt = (int) g[i];
			rt = (int) r[i];
			buffer[i] = bt << 16 | gt << 8 | rt;
		}

		show(buffer, width, height, 1);
	}

	@Override
	public void paintComponent(Graphics g) {
		g.drawImage(image, 0, 0, null);
	}
}
