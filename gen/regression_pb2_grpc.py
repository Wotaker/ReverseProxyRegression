# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import gen.regression_pb2 as regression__pb2


class LinearRegressionStub(object):
    """The Linear Regression service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PosteriorPredictive = channel.unary_unary(
                '/regression.LinearRegression/PosteriorPredictive',
                request_serializer=regression__pb2.RegressionData.SerializeToString,
                response_deserializer=regression__pb2.LRPosterior.FromString,
                )


class LinearRegressionServicer(object):
    """The Linear Regression service definition
    """

    def PosteriorPredictive(self, request, context):
        """Calculates the regression line to the given data

        Returns the posterior parameters as well as predictive posterior parameters
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LinearRegressionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PosteriorPredictive': grpc.unary_unary_rpc_method_handler(
                    servicer.PosteriorPredictive,
                    request_deserializer=regression__pb2.RegressionData.FromString,
                    response_serializer=regression__pb2.LRPosterior.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'regression.LinearRegression', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LinearRegression(object):
    """The Linear Regression service definition
    """

    @staticmethod
    def PosteriorPredictive(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/regression.LinearRegression/PosteriorPredictive',
            regression__pb2.RegressionData.SerializeToString,
            regression__pb2.LRPosterior.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class GaussianProcessStub(object):
    """The Linear Regression service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GaussianProcessRegression = channel.unary_unary(
                '/regression.GaussianProcess/GaussianProcessRegression',
                request_serializer=regression__pb2.RegressionData.SerializeToString,
                response_deserializer=regression__pb2.GPPosterior.FromString,
                )


class GaussianProcessServicer(object):
    """The Linear Regression service definition
    """

    def GaussianProcessRegression(self, request, context):
        """Calculates the regression function to the given data

        Returns the posterior parameters
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GaussianProcessServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GaussianProcessRegression': grpc.unary_unary_rpc_method_handler(
                    servicer.GaussianProcessRegression,
                    request_deserializer=regression__pb2.RegressionData.FromString,
                    response_serializer=regression__pb2.GPPosterior.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'regression.GaussianProcess', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GaussianProcess(object):
    """The Linear Regression service definition
    """

    @staticmethod
    def GaussianProcessRegression(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/regression.GaussianProcess/GaussianProcessRegression',
            regression__pb2.RegressionData.SerializeToString,
            regression__pb2.GPPosterior.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
