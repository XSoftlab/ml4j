package net.xsoftlab.ml4j.model.supervised.svm.libsvm;

public class Problem implements java.io.Serializable {

	private static final long serialVersionUID = 3363256519179239851L;

	public int length;
	public double[] y;
	public Node[][] x;

	public Problem() {
		super();
	}

	public Problem(Node[][] x, double[] y) {
		super();
		this.x = x;
		this.y = y;
		this.length = y.length;
	}
}
