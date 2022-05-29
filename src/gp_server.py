from concurrent import futures
import time
import logging
import numpy as np
from numpy.linalg import inv

import grpc
import gen.regression_pb2 as regression_pb2
import gen.regression_pb2_grpc as regression_pb2_grpc
from gen.regression_pb2 import NumpyArray
from data.dataflow import *


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def sqdist(X, Y):
    '''
    Calculate all-to-all squared distances between points in X and Y.
    
    Args:
        X: Point coordinates,
           shape: n times d.
        Y: Point coordinates,
           shape: m times d.
    
    Returns:
        n times m matrix with squared Euclidean distances.
    '''
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True)
    
    return X2 + Y2.T - 2 * X @ Y.T


def gaussian_kernel(X, Y, l=1.0):
    return np.exp(-(sqdist(X, Y) / 2 * l * l))


def gp_regression(V, U, y, sigma, kernel=gaussian_kernel, **kernel_args):
    
    K_uu = kernel(U, U, **kernel_args)
    K_uv = kernel(U, V, **kernel_args)
    K_vu = kernel(V, U, **kernel_args)
    K_vv = kernel(V, V, **kernel_args)
    
    I = np.diag(np.diag(np.ones_like(K_uu)))

    M = inv(K_uu + sigma**2 * I)
    
    mu_v = K_vu @ M @ y
    Sigma_v = K_vv - K_vu @ M @ K_uv
    
    return mu_v, Sigma_v


def gaussian_process_regression(data, sigma, regression_points):
    '''
    Calculates gaussian process regression in regression_points
    
    Args:
        data:              Input data, shape: (n_points, 2)
        sigma:             Measurement noise, scalar
        regression_points: Points where the evaluation is calculated
        
        
    Returns:
        mean and stds of the regression function in regression_points
    '''
    
    # Prepare data
    U, y = data[:, 0], data[:, 1]
    U, y = np.reshape(U, (-1, 1)), np.reshape(y, (-1, 1))
    
    # Gaussian Process regression
    mu_v, Sigma_v = gp_regression(regression_points, U, y, sigma, kernel=gaussian_kernel, l=1.)
    
    return mu_v, Sigma_v


class GaussianProcess(regression_pb2_grpc.GaussianProcessServicer):

    def GaussianProcessRegression(self, request, context):

        # Log regression request
        print("[Info] Received gaussian process regression request")

        # Deserialize request payload
        serialized_data = request.data
        data = np_deserialize(
            serialized_data.array_bytes, 
            (serialized_data.rows, serialized_data.cols)
        )

        serialized_rp = request.regression_points
        V = np_deserialize(
            serialized_rp.array_bytes, 
            (serialized_rp.rows, serialized_rp.cols)
        )

        sigma = request.sigma

        # Calculate posterior mu_V and Sigma_V with gaussian processes
        mu_V, Sigma_V = gaussian_process_regression(data, sigma, V)

        # Pack result into GPPosterior msg and return it
        ser_mu_V = np_serialize(mu_V)
        posterior_mu_V = regression_pb2.NumpyArray(
            array_bytes=ser_mu_V[0],
            rows=ser_mu_V[1][0],
            cols=ser_mu_V[1][1]
        )

        ser_Sigma_V = np_serialize(Sigma_V)
        posterior_Sigma_V = regression_pb2.NumpyArray(
            array_bytes=ser_Sigma_V[0],
            rows=ser_Sigma_V[1][0],
            cols=ser_Sigma_V[1][1]
        )

        return regression_pb2.GPPosterior(mu_V=posterior_mu_V, cov_V=posterior_Sigma_V)


def serve():
    port = '60061'

    with open(r'/home/wotaker/server.key', 'rb') as f: #path to you key location 
        private_key = f.read()
    with open(r'/home/wotaker/server.crt', 'rb') as f: #path to your cert location
        certificate_chain = f.read()
    server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain,),))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    regression_pb2_grpc.add_GaussianProcessServicer_to_server(GaussianProcess(), server)
    server.add_secure_port('[::]:' + port, server_credentials)
    server.start()

    print("[Info] Server starded successfully, waiting for requests!")

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
