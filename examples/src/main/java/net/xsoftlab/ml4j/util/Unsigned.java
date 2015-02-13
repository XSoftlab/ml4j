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
import java.nio.ByteBuffer;

/**
 * Utility class to get and put unsigned values to a ByteBuffer object. All
 * methods here are static and take a ByteBuffer argument. Since java does not
 * provide unsigned primitive types, each unsigned value read from the buffer is
 * promoted up to the next bigger primitive data type. getUnsignedByte() returns
 * a short, getUnsignedShort() returns an int and getUnsignedInt() returns a
 * long. There is no getUnsignedLong() since there is no primitive type to hold
 * the value returned. If needed, methods returning BigInteger could be
 * implemented. Likewise, the put methods take a value larger than the type they
 * will be assigning. putUnsignedByte takes a short argument, etc.
 * 
 * @author Ron Hitchens (ron@ronsoft.com)
 */
public class Unsigned {
	public static short getUnsignedByte(ByteBuffer bb) {
		return ((short) (bb.get() & 0xff));
	}

	public static void putUnsignedByte(ByteBuffer bb, int value) {
		bb.put((byte) (value & 0xff));
	}

	public static short getUnsignedByte(ByteBuffer bb, int position) {
		return ((short) (bb.get(position) & (short) 0xff));
	}

	public static void putUnsignedByte(ByteBuffer bb, int position, int value) {
		bb.put(position, (byte) (value & 0xff));
	}

	// ---------------------------------------------------------------
	public static int getUnsignedShort(ByteBuffer bb) {
		return (bb.getShort() & 0xffff);
	}

	public static void putUnsignedShort(ByteBuffer bb, int value) {
		bb.putShort((short) (value & 0xffff));
	}

	public static int getUnsignedShort(ByteBuffer bb, int position) {
		return (bb.getShort(position) & 0xffff);
	}

	public static void putUnsignedShort(ByteBuffer bb, int position, int value) {
		bb.putShort(position, (short) (value & 0xffff));
	}

	// ---------------------------------------------------------------
	public static long getUnsignedInt(ByteBuffer bb) {
		return ((long) bb.getInt() & 0xffffffffL);
	}

	public static void putUnsignedInt(ByteBuffer bb, long value) {
		bb.putInt((int) (value & 0xffffffffL));
	}

	public static long getUnsignedInt(ByteBuffer bb, int position) {
		return ((long) bb.getInt(position) & 0xffffffffL);
	}

	public static void putUnsignedInt(ByteBuffer bb, int position, long value) {
		bb.putInt(position, (int) (value & 0xffffffffL));
	}
}