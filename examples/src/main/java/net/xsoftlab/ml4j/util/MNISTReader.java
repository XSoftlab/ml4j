package net.xsoftlab.ml4j.util;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JPanel;

import net.xsoftlab.ml4j.exception.Ml4jException;

import org.jblas.FloatMatrix;

/**
 * 读取MNIST文件
 * 
 * @author 王彦超
 *
 */
public class MNISTReader {

	public static FloatMatrix loadMNISTImages(String filename) throws IOException {

		DataInputStream in = null;
		FloatMatrix matrix = null;
		try {
			in = new DataInputStream(new FileInputStream(filename));

			int magicNumber = in.readInt();
			if (magicNumber != 2051) {
				Ml4jException.logAndThrowException("magic number = " + magicNumber + " 不正确，应为2051");
			}

			int count = in.readInt();
			int rows = in.readInt();
			int colums = in.readInt();

			int size = rows * colums;
			float[] buffer = new float[size];
			matrix = new FloatMatrix(count, size);

			for (int i = 0; i < count; i++) {
				for (int j = 0; j < size; j++) {
					buffer[j] = in.read();
				}
				matrix.putRow(i, new FloatMatrix(buffer));
			}
		} finally {
			if (in != null)
				in.close();
		}

		return matrix;
	}

	public static void main(String[] args) throws Exception {

		long time = System.currentTimeMillis();
		loadMNISTImages("d:/train-images-idx3-ubyte");
		System.out.println((System.currentTimeMillis() - time) / 1000f);
	}

}

class ImagePanel extends JPanel {

	private static final long serialVersionUID = 1107409750852921744L;

	private BufferedImage image;

	public ImagePanel(BufferedImage image) {
		this.image = image;
	}

	public static void show(int[] buffer, int rows, int colums, int count) {
		BufferedImage bufferedImage = new BufferedImage(rows, colums * count, BufferedImage.TYPE_INT_RGB);
		bufferedImage.setRGB(0, 0, rows, colums * count, buffer, 0, rows);
		JFrame jframe = new JFrame();
		jframe.add(new ImagePanel(bufferedImage));
		jframe.setSize(200, 200);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setVisible(true);
	}

	@Override
	public void paintComponent(Graphics g) {
		g.drawImage(image, 0, 0, null);
	}

}
