from types import SimpleNamespace
from typing import Optional
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from typing import Literal


# TODO: for conditional parameters, e.g. numeric: -> two variables u = 1 if disabled and 0 otherwise, v = value if not

class Feedforward:
    def __init__(self, num_hidden_layers: int = 3, num_hidden_units: int = 64,
                 activation: str = 'relu'):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.activation = activation

    def get(self, input_length: int, output_length: int):
        network = keras.Sequential()
        network.add(keras.Input(shape=(input_length, )))
        for _ in range(self.num_hidden_layers):
            network.add(keras.layers.Dense(units=self.num_hidden_units, activation=self.activation))
        network.add(keras.layers.Dense(units=output_length))
        network(keras.Input(shape=(input_length, )))
        return network


class ConcatenateForward:
    def __init__(self, feedforward_network: Feedforward = Feedforward(num_hidden_layers=3)):
        self.feedforward_network = feedforward_network

    # TODO: type annotations for inputs
    def get(self, configuration_encoder_output, instance_encoder_output):
        concatenate_layer = keras.layers.concatenate(
            [configuration_encoder_output, instance_encoder_output]
        )
        feedforward_model = self.feedforward_network.get(
            configuration_encoder_output.shape[1] + instance_encoder_output.shape[1],
            1
        )
        return keras.Model(
            inputs=[configuration_encoder_output, instance_encoder_output],
            outputs=feedforward_model(concatenate_layer)
        )


class RadialPrediction(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.math.reciprocal(1.0 + tf.square(tf.norm(tf.subtract(*inputs), axis=-1)))


class OverlayingForward:
    def __init__(self):
        pass
    # TODO: type annotations for inputs
    def get(self, configuration_encoder_output, instance_encoder_output):
        squared_distance_layer = RadialPrediction()(
            [configuration_encoder_output, instance_encoder_output]
        )
        return keras.Model(
            inputs=[configuration_encoder_output, instance_encoder_output],
            outputs=squared_distance_layer
        )


class Losses:
    def __init__(self, performance_loss: str = "mse", feature_loss: str = "mse", parameter_loss: Optional[str] = None,
                 performance_loss_weight: float = 10.0, feature_loss_weight: float = 1.0, parameter_loss_weight: Optional[float] = None):
        self.performance_loss = performance_loss
        self.feature_loss = feature_loss
        self.parameter_loss = parameter_loss
        self.performance_loss_weight = performance_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.parameter_loss_weight = parameter_loss_weight

    def get(self, num_projections: Literal['one', 'two']):
        if num_projections == 'one':
            losses = {'performance': self.performance_loss, 'feature': self.feature_loss}
            loss_weights = {'performance': self.performance_loss_weight, 'feature': self.feature_loss_weight}
        elif num_projections == 'two':
            parameter_loss = 'mse' if self.parameter_loss is None else self.parameter_loss
            parameter_loss_weight = 0.0 if self.parameter_loss_weight is None else self.parameter_loss_weight
            losses = {'performance': self.performance_loss, 'feature': self.feature_loss, 'parameter': parameter_loss}
            loss_weights = {'performance': self.performance_loss_weight, 'feature': self.feature_loss_weight, 'parameter': parameter_loss_weight}
        else:
            assert False, 'Invalid num_projections (must be one of: one, two)'
        return SimpleNamespace(losses=losses, loss_weights=loss_weights)


class Projector:
    def __init__(self,
                 instance_encoder: Feedforward = Feedforward(num_hidden_layers=0),
                 instance_decoder: Feedforward = Feedforward(num_hidden_layers=0),
                 configuration_encoder: Optional[Feedforward] = Feedforward(num_hidden_layers=3),
                 configuration_decoder: Optional[Feedforward] = None,
                 performance_model: ConcatenateForward = ConcatenateForward(),
                 optimizer: str = "Adam",
                 loss: Optional[Losses] = Losses(performance_loss_weight=10.0, feature_loss_weight=1.0),
                 num_projections: Literal['one', 'two'] = 'two'
                 ):
        self.instance_encoder = instance_encoder
        self.instance_decoder = instance_decoder
        self.configuration_encoder = configuration_encoder
        self.configuration_decoder = configuration_decoder
        self.performance_model = performance_model
        self.optimizer = optimizer
        self.loss = loss
        self.num_projections = num_projections
    def fit(self,
            preprocessed_features: np.ndarray,
            preprocessed_parameters: np.ndarray,
            preprocessed_performance: pd.DataFrame,
            batch_size: int = 32,
            epochs: int = 150,
            validation_split: float = 0.1,
            verbose='auto'
            ):
        num_features = preprocessed_features.shape[1]
        num_parameters = preprocessed_parameters.shape[1]
        assert self.num_projections in ['one', 'two']
        if self.configuration_encoder is not None or self.configuration_decoder is not None:
            assert self.num_projections == 'two', 'number of projections is inconsistent with configuration encoder and decoder'
        # num_projections = 'one' if self.configuration_encoder is None else 'two'
        long_format_feature_matrix, long_format_parameter_matrix, preprocessed_performance_vector \
            = get_long_format_metadata(preprocessed_features, preprocessed_parameters, preprocessed_performance)
        # instance encoder
        instance_encoder = self.instance_encoder.get(input_length=num_features, output_length=2)
        instance_encoder_input = instance_encoder.input
        instance_encoder_output = instance_encoder(instance_encoder_input)
        # instance decoder
        instance_decoder = self.instance_decoder.get(input_length=2, output_length=num_features)
        instance_decoder_input = instance_encoder_output
        instance_decoder_output = instance_decoder(instance_decoder_input)
        instance_decoder._name = 'feature'
        if self.num_projections == 'two':
            # configuration encoder
            self.configuration_encoder = self.configuration_encoder if self.configuration_encoder is not None \
                else Feedforward(num_hidden_layers=3)
            configuration_encoder = self.configuration_encoder.get(input_length=num_parameters, output_length=2)
            configuration_encoder_input = configuration_encoder.input
            configuration_encoder_output = configuration_encoder(configuration_encoder_input)
            # configuration decoder
            self.configuration_decoder = self.configuration_decoder if self.configuration_decoder is not None \
                else Feedforward(num_hidden_layers=3)
            configuration_decoder = self.configuration_decoder.get(input_length=2, output_length=num_parameters)
            configuration_decoder_input = configuration_encoder_output
            configuration_decoder_output = configuration_decoder(configuration_decoder_input)
        else:
            configuration_encoder = None
            configuration_encoder_input = keras.Input(shape=(num_parameters, ))
            configuration_encoder_output = configuration_encoder_input
            configuration_decoder = None
            configuration_decoder_input = configuration_encoder_output
            configuration_decoder_output = configuration_decoder_input

        performance_model = self.performance_model.get(configuration_encoder_output, instance_encoder_output)
        performance_model_output = performance_model([configuration_encoder_output, instance_encoder_output])

        performance_model_output = keras.layers.Lambda(lambda x: x, name='performance')(performance_model_output)
        instance_decoder_output = keras.layers.Lambda(lambda x: x, name='feature')(instance_decoder_output)


        if self.num_projections == 'two':
            configuration_decoder_output = keras.layers.Lambda(lambda x: x, name='parameter')(configuration_decoder_output)
            outputs = [
                performance_model_output,
                instance_decoder_output,
                configuration_decoder_output
            ]
            fit_y = [preprocessed_performance_vector, long_format_feature_matrix, long_format_parameter_matrix]
        else:
            outputs = [
                performance_model_output,
                instance_decoder_output,
            ]
            fit_y = [preprocessed_performance_vector, long_format_feature_matrix]

        model = keras.Model(
            inputs=[instance_encoder_input, configuration_encoder_input],
            outputs=outputs
        )

        loss_object_out = self.loss.get(self.num_projections)

        model.compile(
            optimizer=self.optimizer, loss=loss_object_out.losses, loss_weights=loss_object_out.loss_weights
        )

        history = model.fit(
            [long_format_feature_matrix, long_format_parameter_matrix], fit_y,
            batch_size=batch_size, epochs=epochs, validation_split=validation_split,
            verbose=verbose
        )

        return SimpleNamespace(
            model=model,
            instance_encoder=instance_encoder,
            instance_decoder=instance_decoder,
            configuration_encoder=configuration_encoder,
            configuration_decoder=configuration_decoder,
            history=history
        )


# Get the (features, parameters, value) evaluation vectors
def get_long_format_metadata(preprocessed_features, preprocessed_parameters, preprocessed_performance):
    long_format_feature_matrix = preprocessed_features[
        preprocessed_performance.iloc[:, 0].astype(np.int64)]
    long_format_parameter_matrix = preprocessed_parameters[
        preprocessed_performance.iloc[:, 1].astype(np.int64)]
    preprocessed_performance_vector = preprocessed_performance.iloc[:, 2].to_numpy()
    return long_format_feature_matrix, long_format_parameter_matrix, preprocessed_performance_vector
