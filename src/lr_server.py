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


def bayes_lin_reg(prior_mu, prior_Sigma, sigma, X, y):
    sigma2_inv = 1 / (sigma**2)
    posterior_Sigma = inv(inv(prior_Sigma) + sigma2_inv * (np.transpose(X) @ X))
    posterior_mu = posterior_Sigma @ (sigma2_inv * (np.transpose(X) @ y) + inv(prior_Sigma) @ prior_mu)
    
    return posterior_mu, posterior_Sigma


def posterior_predictive(posterior_mu, posterior_Sigma, sigma, x):
    # x = np.transpose(x)
    mu_y = np.squeeze(np.transpose(posterior_mu) @ x)
    std_y = (sigma**2) + np.transpose(x) @ posterior_Sigma @ x
    return mu_y, np.sqrt(std_y)


def linear_regression(data, sigma, regression_points):
    '''
    Calculates predictive posterior in regression_points
    
    Args:
        data:              Input data, shape: (n_samples, 2)
        sigma:             Input data variance, scalar
        prior_mu:          Prior of the linear regression mean
        prior_Sigma:       Prior of the linear regression covariance
        regression_points: Points where the evaluation is calculated
        
    Returns:
        mean and covariance of the regression slope, means and stds of predictions
    '''
    
    # Prepare data
    x, y = data[:, 0], data[:, 1]
    x, y = np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((x, ones), axis=1)
    
    # Prepare prior
    y_mean, y_var = np.mean(data[:, 0]), np.var(data[:, 0])
    mu_0 = np.array([0, y_mean]).reshape(-1, 1)
    Sigma_0 = np.array([[1, 0],
                        [0, y_var]])
    
    # Prepare regression points
    regression_points = np.reshape(regression_points, (-1, 1))
    regression_points = np.concatenate((regression_points, np.ones_like(regression_points)), 1)
    
    # Bayesian regression
    mu_n, Sigma_n = bayes_lin_reg(mu_0, Sigma_0, sigma, x, y)
    mu_n = np.squeeze(mu_n)
    
    # Posterior predictive
    no_tests = regression_points.shape[0] 
    y_mu, y_sigma = np.empty(no_tests), np.empty(no_tests)
    for idx, tx in enumerate(regression_points):
        y_mu[idx], y_sigma[idx] = posterior_predictive(mu_n, Sigma_n, sigma, tx)
    
    return mu_n, Sigma_n, y_mu, y_sigma


class LinearRegression(regression_pb2_grpc.LinearRegressionServicer):

    def PosteriorPredictive(self, request, context):
        
        # Log regression request
        print("[Info] Received linear regression request")

        # Deserialize request payload
        serialized_data = request.data
        data = np_deserialize(
            serialized_data.array_bytes, 
            (serialized_data.rows, serialized_data.cols)
        )

        serialized_rp = request.regression_points
        regression_points = np_deserialize(
            serialized_rp.array_bytes, 
            (serialized_rp.rows, serialized_rp.cols)
        )

        sigma = request.sigma

        # Calculate posteriors
        mu_n, Sigma_n, y_mu, y_sigma = linear_regression(data, sigma, regression_points)

        # Pack result into LRPosterior msg and return it
        ser_mu_n = np_serialize(mu_n)
        posterior_mu_n = regression_pb2.NumpyArray(
            array_bytes=ser_mu_n[0],
            rows=ser_mu_n[1][0],
            cols=0
        )

        ser_Sigma_n = np_serialize(Sigma_n)
        posterior_Sigma_n = regression_pb2.NumpyArray(
            array_bytes=ser_Sigma_n[0],
            rows=ser_Sigma_n[1][0],
            cols=ser_Sigma_n[1][1]
        )

        ser_y_mu = np_serialize(y_mu)
        posterior_y_mu = regression_pb2.NumpyArray(
            array_bytes=ser_y_mu[0],
            rows=ser_y_mu[1][0],
            cols=0
        )

        ser_y_sigma = np_serialize(y_sigma)
        posterior_y_sigma = regression_pb2.NumpyArray(
            array_bytes=ser_y_sigma[0],
            rows=ser_y_sigma[1][0],
            cols=0
        )

        return regression_pb2.LRPosterior(
            mu_n=posterior_mu_n,
            cov_n=posterior_Sigma_n,
            y_mu=posterior_y_mu,
            y_std=posterior_y_sigma
        )


def serve():
    port = '60062'

    with open(r'/home/wotaker/server.key', 'rb') as f: #path to you key location 
        private_key = f.read()
    with open(r'/home/wotaker/server.crt', 'rb') as f: #path to your cert location
        certificate_chain = f.read()
    server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain,),))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    regression_pb2_grpc.add_LinearRegressionServicer_to_server(LinearRegression(), server)
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
