package net.xsoftlab.ml4j.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestUtil {

	public static long time;

	public static Logger logger = LoggerFactory.getLogger(TestUtil.class);

	public static final String RESOURCES_PATH = System.getProperty("user.dir") + "/resources";
	public static final String COURSE_ML_PATH = RESOURCES_PATH + "/coursera/ml";
	public static final String UFLDL_NEW_PATH = RESOURCES_PATH + "/ufldl/newVersion";

	public static void tic() {
		time = System.currentTimeMillis();
	}

	public static void toc() {
		int h, m, s;
		long useTime = (System.currentTimeMillis() - time) / 1000;
		h = (int) (useTime / 3600);
		m = (int) (useTime / 60);
		s = (int) (useTime % 60);
		logger.info("用时: {} 小时, {} 分钟, {} 秒.\n", new Object[] { h, m, s });
	}
}
