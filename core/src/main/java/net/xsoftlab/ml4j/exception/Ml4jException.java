package net.xsoftlab.ml4j.exception;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Ml4jException extends RuntimeException {

	private static final long serialVersionUID = 5254527327442579657L;

	public static Logger logger = LoggerFactory.getLogger(Ml4jException.class);

	public Ml4jException(String message) {

		super(message);
	}

	public static void logAndThrowException(String message) {
		Ml4jException e = new Ml4jException(message);
		logger.error(message, e);
		throw e;
	}
}
