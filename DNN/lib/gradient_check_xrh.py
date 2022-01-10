#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


def rel_error(grad_num, grad):
    """
    计算 通过数值方法得到的梯度 和 通过反向传播得到的梯度的距离

    :param x: 通过数值方法得到的梯度
    :param y: 通过反向传播得到的梯度
    :return:
    """
    return np.max(np.abs(grad_num - grad) / (np.maximum(1e-8, np.abs(grad_num) + np.abs(grad))))

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    返回 数值梯度


    :param f: f should be a function that takes a single argument
    :param x: x is the point (numpy array) to evaluate the gradient at
    :param verbose:
    :param h:
    :return:
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    返回 数值梯度, 要考虑下一层传递给当前层的梯度

    :param f: 当前层的前向传播算法
    :param x: 目标数组
    :param df: 下一层传给当前层的梯度
    :param h: 一个很小的数
    :return:
    """

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # 遍历数组中的每一个元素(每一个位)
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)  # 梯度检验的公式为 用数值方法估计损失函数对参数的某一位的偏导数
        # 这里转换为： 下一层的梯度(损失函数对当前层输出的偏导数)传给当前层
        it.iternext()
    return grad


class GradientCheck:
    """
    梯度检验

    ref:
    deeplearning.ai 吴恩达 -> 第二课第一周编程作业/assignment1

    Author: xrh
    Date: 2021-08-10

    """

    def dictionary_to_vector(self,parameters):
        """
        将模型的所有参数拍平后放到一个大的向量 theta中

        :param parameters:
        :return: theta
        """

        keys = []
        count = 0
        for key in parameters.keys():

            # flatten parameter
            new_vector = np.reshape(parameters[key], (-1, 1))
            keys = keys + [key] * new_vector.shape[0]

            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys


    def vector_to_dictionary(self,theta,parameters):
        """
        将大的向量 theta 转换为模型的参数 parameters

        :param theta:
        :param parameters:
        :return:
        """

        new_parameters = {}

        start = 0

        for param_name,param_value in  parameters.items():

            param_shape = np.shape(param_value)
            offset = np.prod(param_shape) # 对列表中的所有元素求乘积

            new_parameters[param_name] = theta[start:start+offset].reshape(param_shape)

            start += offset

        return new_parameters


    def gradients_to_vector(self,gradients):
        """
        将模型的所有参数的梯度拍平后放到一个大的向量 theta中

        :param gradients:
        :return: theta
        """

        count = 0
        for key in gradients.keys():
            # flatten parameter
            new_vector = np.reshape(gradients[key], (-1, 1))

            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta


    def gradient_check(self,gradients,forward_propagation,parameters, epsilon=1e-7, **kwargs):
        """
        梯度检测, 检查反向传播算法求得的梯度是否正确

        :param gradients: 向传播算法求得的梯度, 其中参数的顺序必须与 parameters 保持一致
        :param forward_propagation: 正向传播算法
        :param parameters: 模型的参数
        :param epsilon: 一个很小的数

        :param **kwargs: 用于传入 forward_propagation 的其他参数, 使用 key-value 的格式

        :return: difference : 估计的梯度 和 反向传播求得的梯度 的差值

        """

        # Set-up variables
        parameters_values, _ = self.dictionary_to_vector(parameters)

        grad = self.gradients_to_vector(gradients)

        num_parameters = parameters_values.shape[0]

        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        # Compute gradapprox
        for i in range(num_parameters):

            # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
            # "_" is used because the function you have to outputs two parameters but we only care about the first one

            thetaplus = np.copy(parameters_values)  # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2

            params = self.vector_to_dictionary(thetaplus,parameters)

            J_plus[i], _ = forward_propagation(parameters=params,**kwargs)  # Step 3

            # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
            thetaminus = np.copy(parameters_values)  # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2

            J_minus[i], _ = forward_propagation( parameters=self.vector_to_dictionary(thetaminus,parameters),**kwargs)  # Step 3

            # Compute gradapprox[i]
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)

        # Compare gradapprox to backward propagation gradients by computing difference.

        numerator = np.linalg.norm(grad) - np.linalg.norm(gradapprox) #TODO: 此处与公式不符  # Step 1'

        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
        difference = numerator / denominator  # Step 3'

        if difference > 1e-7:
            print(
               "There is a mistake in the backward propagation! difference = " + str(difference) )
        else:
            print(
                "Your backward propagation works perfectly fine! difference = " + str(difference) )

        return difference

class UnitTest:
    """
    单元测试

    """
    def test_gradient_check(self):

        def sigmoid(x):
            """
            Compute the sigmoid of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(x)
            """
            s = 1 / (1 + np.exp(-x))
            return s

        def relu(x):
            """
            Compute the relu of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- relu(x)
            """
            s = np.maximum(0, x)

            return s

        def forward(parameters,X, Y):
            """
            Implements the forward propagation (and computes the cost) presented in Figure 3.

            Arguments:
            X -- training set for m examples
            Y -- labels for m examples
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape (5, 4)
                            b1 -- bias vector of shape (5, 1)
                            W2 -- weight matrix of shape (3, 5)
                            b2 -- bias vector of shape (3, 1)
                            W3 -- weight matrix of shape (1, 3)
                            b3 -- bias vector of shape (1, 1)

            Returns:
            cost -- the cost function (logistic cost for one example)
            """

            # retrieve parameters
            m = X.shape[1]
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            Z1 = np.dot(W1, X) + b1
            A1 = relu(Z1)
            Z2 = np.dot(W2, A1) + b2
            A2 = relu(Z2)
            Z3 = np.dot(W3, A2) + b3
            A3 = sigmoid(Z3)

            # Cost
            logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
            cost = 1. / m * np.sum(logprobs)

            cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

            return cost, cache

        def backward(X, Y, cache):
            """
            Implement the backward propagation presented in figure 2.

            Arguments:
            X -- input datapoint, of shape (input size, 1)
            Y -- true "label"
            cache -- cache output from forward_propagation()

            Returns:
            gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
            """

            m = X.shape[1]
            (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

            dZ3 = A3 - Y
            dW3 = 1. / m * np.dot(dZ3, A2.T)
            db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.dot(W3.T, dZ3)
            dZ2 = np.multiply(dA2, np.int64(A2 > 0))
            dW2 = 1. / m * np.dot(dZ2, A1.T)
            db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(W2.T, dZ2)
            dZ1 = np.multiply(dA1, np.int64(A1 > 0))
            dW1 = 1. / m * np.dot(dZ1, X.T)
            db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

            # gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
            #              "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
            #              "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

            gradients = {
                          "dW1": dW1,
                          "db1": db1,
                          "dW2": dW2,
                          "db2": db2,
                          "dW3": dW3,
                          "db3": db3
                        }

            return gradients

        def gradient_check_n_test_case():

            np.random.seed(1)
            x = np.random.randn(4, 3)
            y = np.array([1, 1, 0])
            W1 = np.random.randn(5, 4)
            b1 = np.random.randn(5, 1)
            W2 = np.random.randn(3, 5)
            b2 = np.random.randn(3, 1)
            W3 = np.random.randn(1, 3)
            b3 = np.random.randn(1, 1)

            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2,
                          "W3": W3,
                          "b3": b3}

            return x, y, parameters

        X, Y, parameters = gradient_check_n_test_case()

        cost, cache = forward(parameters,X, Y)
        gradients = backward(X, Y, cache)

        gc = GradientCheck()

        difference = gc.gradient_check(gradients,forward, parameters, X=X, Y=Y)


if __name__ == '__main__':

    test = UnitTest()

    test.test_gradient_check()


