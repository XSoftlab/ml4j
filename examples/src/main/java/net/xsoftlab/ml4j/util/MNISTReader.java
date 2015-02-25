package net.xsoftlab.ml4j.util;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

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
public class MNISTReader extends TestUtil {

	public static void main(String[] args) throws Exception {

		tic();
		loadMNISTImages("d:/train-images-idx3-ubyte");
		toc();
	}

	public static void loadMNISTImages(String filePath) throws IOException {

		FileInputStream in = null;
		FloatMatrix matrix = null;
		try {
			in = new FileInputStream(filePath);
			// 获取输入输出通道
			FileChannel channel = in.getChannel();
			ByteBuffer byteBuffer = ByteBuffer.allocate(16);
			channel.read(byteBuffer);

			byteBuffer.rewind();
			int magicNumber = byteBuffer.getInt();
			if (magicNumber != 2051) {
				Ml4jException.logAndThrowException("magic number = " + magicNumber + " 不正确，应为2051");
			}

			int count = byteBuffer.getInt();
			int rows = byteBuffer.getInt();
			int colums = byteBuffer.getInt();

			int size = rows * colums;
			byteBuffer = ByteBuffer.allocate(size);
			float[] buffer = new float[size];
			matrix = new FloatMatrix(count, size);

			int j = 0;
			for (int i = 0; i < count; i++) {
				j = 0;
				byteBuffer.clear();
				channel.read(byteBuffer);
				byteBuffer.rewind();
				while (byteBuffer.hasRemaining()) {
					buffer[j++] = byteBuffer.get() & 0xff;
				}
				matrix.putRow(i, new FloatMatrix(buffer));
			}

			ImagePanel.show(matrix.getRow(10000).data, rows, colums, 1);
		} finally {
			if (in != null)
				in.close();
		}

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

	public static void show(float[] buffer, int rows, int colums, int count) {

		int[] iBuffer = new int[buffer.length];
		for (int i = 0; i < buffer.length; i++)
			iBuffer[i] = (int) buffer[i];

		BufferedImage bufferedImage = new BufferedImage(rows, colums * count, BufferedImage.TYPE_INT_RGB);
		bufferedImage.setRGB(0, 0, rows, colums * count, iBuffer, 0, rows);
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