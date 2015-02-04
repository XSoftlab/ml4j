package net.xsoftlab.ml4j.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestUtil {

	public static long time;

	public static Logger logger = LoggerFactory.getLogger(TestUtil.class);

	public static void tic() {
		time = System.currentTimeMillis();
	}

	public static void toc() {

		logger.debug("UseTime {} ms", (System.currentTimeMillis() - time));
	}
}
