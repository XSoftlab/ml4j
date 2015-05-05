package net.xsoftlab.ml4j.minfunc;

import net.xsoftlab.ml4j.model.supervised.BaseModel;

import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 最优化函数
 * 
 * @author 王彦超
 *
 */
public abstract class MinFunc {

	protected BaseModel model;// 训练模型
	protected int maxIter = 500;// 最大训练次数
	protected float epsilon = (float) 1e-6;// 精度阈值

	protected FloatMatrix theta;// 参数
	protected boolean logFlag = true;// 是否打印过程日志

	Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * 最优化theta
	 * 
	 * @return theta
	 */
	public abstract FloatMatrix train();

	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;
	}

	public void setTheta(FloatMatrix theta) {
		this.theta = theta;
	}

	public void setCostFlag(boolean costFlag) {
		this.logFlag = costFlag;
	}
}
