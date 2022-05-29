from __future__ import print_function
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import uniform

import grpc
import gen.regression_pb2 as regression_pb2
import gen.regression_pb2_grpc as regression_pb2_grpc
from gen.regression_pb2 import NumpyArray
from data.dataflow import *

np.random.seed(1234)
plt.rcParams["figure.figsize"] = [16, 9]


TYPE_LINEAR = 'linear'
TYPE_GAUSSIAN = 'gaussian'


def call_gp_servis(
    stub: regression_pb2_grpc.GaussianProcessStub, 
    regression_data: regression_pb2.RegressionData, 
    data: np.ndarray, rp: np.ndarray
):

    # Remote call
    response = stub.GaussianProcessRegression(regression_data)

    # Deserialize response
    ser_mu_V = response.mu_V
    mu_V = np_deserialize(ser_mu_V.array_bytes, (ser_mu_V.rows, ser_mu_V.cols))

    ser_cov_V = response.cov_V
    cov_V = np_deserialize(ser_cov_V.array_bytes, (ser_cov_V.rows, ser_cov_V.cols))

    # Process response
    print(f"\nmu_V:\n{mu_V}\n\ncov_V:\n{cov_V}")

    # Plot results
    _, axes = plt.subplots(figsize=(14, 7))
    plot_gpr(
        axes, rp, mu_V, cov_V, 
        np.reshape(data[:, 0], (-1, 1)), 
        np.reshape(data[:, 1], (-1, 1))
    )
    plt.show()

    # Save results
    np_to_csv(mu_V, "results/gp_predictions_mean.csv")
    np_to_csv(cov_V, "results/gp_predictions_covariance.csv")



def call_lr_servis(
    stub: regression_pb2_grpc.LinearRegressionStub, 
    regression_data: regression_pb2.RegressionData,
    data: np.ndarray, rp: np.ndarray
):
    # Remote call
    response = stub.PosteriorPredictive(regression_data)

    # Deserialize response
    ser_mu_n = response.mu_n
    mu_n = np_deserialize(ser_mu_n.array_bytes, ser_mu_n.rows)

    ser_cov_n = response.cov_n
    cov_n = np_deserialize(ser_cov_n.array_bytes, (ser_cov_n.rows, ser_cov_n.cols))

    ser_y_mu = response.y_mu
    y_mu = np_deserialize(ser_y_mu.array_bytes, ser_y_mu.rows)

    ser_y_std = response.y_std
    y_std = np_deserialize(ser_y_std.array_bytes, ser_y_std.rows)

    # Process response
    print(f"\nmu_n:\n{mu_n}\n\ncov_n:\n{cov_n}\n\ny_mu:\n{y_mu}\n\ny_std:\n{y_std}")

    # Plot results
    _ = plt.figure(figsize=(14, 7))
    ax = plt.gca()
    plot_fit(ax, mu_n, data, sigma,
            test_x=rp, test_y=y_mu, test_ys=y_std,
            title='Mean fit & posterior predictives')
    plt.show()

    # Save results
    np_to_csv(y_mu, "results/lr_predictions_means.csv")
    np_to_csv(y_std, "results/lr_predictions_uncertainties.csv")


def run(regression_type, data, regression_points, sigma):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    host = 'localhost'
    port = 60060
    with open(r'/home/wotaker/server.crt', 'rb') as f: # path to you cert location
        trusted_certs = f.read()

    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    channel = grpc.secure_channel(f'{host}:{port}', credentials)

    ser_data = np_serialize(data)
    proto_data = regression_pb2.NumpyArray(
        array_bytes=ser_data[0],
        rows=ser_data[1][0],
        cols=ser_data[1][1]
    )

    ser_rp = np_serialize(regression_points)
    proto_rp = regression_pb2.NumpyArray(
        array_bytes=ser_rp[0],
        rows=ser_rp[1][0],
        cols=ser_rp[1][1]
    )

    regression_data = regression_pb2.RegressionData(
        data=proto_data,
        sigma=sigma,
        regression_points=proto_rp
    )
    
    if regression_type == TYPE_GAUSSIAN:
        call_gp_servis(
            regression_pb2_grpc.GaussianProcessStub(channel), 
            regression_data, data, regression_points
        )
    else:
        call_lr_servis(
            regression_pb2_grpc.LinearRegressionStub(channel), 
            regression_data, data, regression_points
        )


if __name__ == "__main__":

    logging.basicConfig()

    # Verify nr of args
    if len(sys.argv) != 5:
        print("Invalid Arguments! This program requires exactly 4 arguments.\nTry to run with " +\
            "`python regression_client.py <regression_type> <path_to_data> <path_to_regression_points> <sigma>`")
        exit(1)
    
    # Load arguments
    regression_type = sys.argv[1].lower()
    assert (regression_type == TYPE_LINEAR) or (regression_type == TYPE_GAUSSIAN), \
        f"Invalid regression type! Choose from [{TYPE_LINEAR}, {TYPE_GAUSSIAN}]"
    path_to_data = sys.argv[2]
    path_to_rp = sys.argv[3]
    sigma = float(sys.argv[4])

    # Load numpy arrays
    try:
        data = np_read_csv(path_to_data)
        regression_points = np_read_csv(path_to_rp)
    except:
        print("LoadError! Unable to load data!")
        exit(1)
    
    # Run client
    run(regression_type, data, regression_points, sigma)
