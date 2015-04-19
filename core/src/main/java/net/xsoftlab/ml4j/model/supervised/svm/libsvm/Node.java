package net.xsoftlab.ml4j.model.supervised.svm.libsvm;

import java.io.Serializable;

public class Node implements Serializable {

	private static final long serialVersionUID = 8345455143219019332L;

	public int index;
	public double value;

	public Node() {
		super();
	}

	public Node(int index, double value) {
		super();
		this.index = index;
		this.value = value;
	}
}
