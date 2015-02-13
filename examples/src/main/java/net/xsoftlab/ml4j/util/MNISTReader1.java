package net.xsoftlab.ml4j.util;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import net.xsoftlab.ml4j.exception.Ml4jException;

import org.jblas.FloatMatrix;

/**
 * 读取MNIST文件
 * 
 * @author 王彦超
 *
 */
public class MNISTReader1 {

	public static void main(String[] args) throws Exception {

		long time = System.currentTimeMillis();
		loadMNISTImages("d:/train-images-idx3-ubyte");
		System.out.println((System.currentTimeMillis() - time) / 1000f);
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
			byteBuffer = ByteBuffer.allocate(size * 4);
			float[] buffer = new float[size];
			matrix = new FloatMatrix(count, size);

			int j = 0;
			for (int i = 0; i < count; i++) {
				j = 0;
				byteBuffer.clear();
				channel.read(byteBuffer);
				byteBuffer.rewind();
				while (byteBuffer.hasRemaining()) {
					buffer[j++] = byteBuffer.getInt();
				}
				matrix.putRow(i, new FloatMatrix(buffer));
			}

		} finally {
			if (in != null)
				in.close();
		}
		
		System.out.println(matrix);
	}
}