#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size 1*m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     * 
     * X: (m*n)
     * y: (1*m)
     * theta: (n*k)
     */

    int iters = (m + batch - 1) / batch;
    for (int iter = 0; iter < iters; iter++) {
        const float *x = &X[iter * batch * n];
        const unsigned char *yy = &y[iter * batch];

        // Z = x @ theta
        // x:(batch*n), theta:(n*k), Z:(batch*k)
        float *Z = new float[batch * k];
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < k; j++) 
            {
                Z[i * k + j] = 0;
                for (int s = 0; s < n; s++)
                    Z[i * k + j] += x[i * n + s] * theta[s * k + j];
            }

        // Z = normalise(exp(Z))
        for (int i = 0; i < batch * k; i++) Z[i] = exp(Z[i]);
        for (int i = 0; i < batch; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) sum += Z[i * k + j];
            for (int j = 0; j < k; j++) Z[i * k + j] /= sum; // row-wise normalization
        }

        // Z = Z - I
        for (int i = 0; i < batch; i++) Z[i * k + yy[i]] -= 1; // minus one-hot vector

        // x.T
        float *x_T = new float[n * batch];
        for (int i = 0; i < batch; i++) 
            for (int j = 0; j < n; j++) 
                x_T[j * batch + i] = x[i * n + j];
        
        // grad = x.T @ Z
        // grad: (n*k), x_T: (n*batch), Z:(batch*k)
        float *grad = new float[n * k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) 
            {
                grad[i * k + j] = 0;
                for (int s = 0; s < batch; s++)
                    grad[i * k + j] += x_T[i * batch + s] * Z[s * k + j];
            }
        
        // update theta
        for (int i = 0; i < n * k; i++) theta[i] -= lr / batch * grad[i];

        delete[] Z;
        delete[] x_T;
        delete[] grad;
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module.
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
