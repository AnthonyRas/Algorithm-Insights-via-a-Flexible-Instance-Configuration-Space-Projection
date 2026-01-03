import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Optional, Literal
from preprocessor import Preprocessor, long_to_wide
from projector import Projector, Feedforward, OverlayingForward, ConcatenateForward, Losses
from plotter import Plotter, MainPlot
import os
import tensorflow as tf
import keras
import random


def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    random.seed(seed)


class ICSA:
    """

    Parameters:

        optimisation_direction: :class:`Literal['min', 'max']` --
            The direction of optimisation.

        mode: :class:`Literal['concatenate', 'overlay']`, **default='concatenate'** --
            A *primary* parameter that determines the defaults of the other parameters, depending on whether the instance/configuration coordinate vectors are concatenated or overlayed in predicting performance

        feature_preprocessing: :class:`Literal['unscaled', 'encode and scale']`, **default='unscaled'** --
            The preprocessing method used for the instance features

        parameter_preprocessing: :class:`Literal['unscaled', 'encode and scale']`, **default='encode and scale'** --
            The preprocessing method used for the algorithm parameters

        performance_preprocessing: :class:`Literal['minmax scale rows', 'unscaled']`, **default=None** --
            The preprocessing method used for the performance metric

        instance_encoder_num_hidden_layers: :class:`int`, **default=None** --
            The number of hidden layers in the instance encoder. If mode='concatenate' then default=0 else if mode='overlay' then default=3.

        instance_encoder_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the instance encoder

        instance_encoder_activation: :class:`str`, **default='relu'** --
            The activation function used in the instance encoder

        instance_decoder_num_hidden_layers: :class:`int`, **default=0** --
            The number of hidden layers in the instance decoder

        instance_decoder_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the instance decoder

        instance_decoder_activation: :class:`str`, **default='relu'** --
            The activation function used in the instance decoder

        configuration_encoder_num_hidden_layers: :class:`int`, **default=3** --
            Number of hidden layers in the configuration encoder

        configuration_encoder_num_hidden_units: :class:`int`, **default=64** --
            Number of hidden units in each layer of the configuration encoder

        configuration_encoder_activation: :class:`str`, **default='relu'** --
            The activation function used in the configuration encoder

        performance_num_hidden_layers: :class:`int`, **default=3** --
            The number of hidden layers in the performance model, if mode == 'concatenate'

        performance_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the performance model, if mode == 'concatenate'

        performance_activation: :class:`str`, **default='relu'** --
            The activation function used in the performance model, if mode == 'concatenate'.

        optimizer: :class:`str`, **default='Adam'** --
            The optimisation algorithm for training the model

        performance_loss_weight: :class:`float`, **default=10.0** --
            The weight of the performance loss in the loss function.

        feature_loss_weight: :class:`float`, **default=1.0** --
            The weight of the feature loss in the loss function

        num_projections: :class:`Literal['one', 'two']`, **default='two'** --
            Whether '1' == project only instance space or '2' == project both instance space and configuration space.

        focus_size: :class:`int`, **default=10** --
            The size of points in the focused space in the ICSA visualization.

        unfocus_size: :class:`int`, **default=7** --
            The size of points in the unfocused space in the ICSA visualization

        instance_colorscale: :class:`str`, **default='Viridis'** --
            The color scale used for instance attributes in the ICSA visualization

        instance_symbol: :class:`str`, **default='circle'** --
            The symbol used to represent instances in the ICSA visualization.

        configuration_colorscale: :class:`str`, **default='Plasma'** --
            The color scale used for configuration attributes in the ICSA visualization

        configuration_symbol: :class:`str`, **default='triangle-up'** --
            The symbol used to represent configurations in the ICSA visualization.

        performance_metric_name: :class:`str`, **default='y'** --
            The name of the performance metric.

        min_height: :class:`int`, **default=750** --
            The minimum height of the ICSA visualization window.

        min_width: :class:`int`, **default=750** --
            The minimum width of the ICSA visualization window.

        text_x: :class:`float`, **default=0.025** --
            The x-coordinate for the positioning of text, labelling categories, in the ICSA visualization

        text_y: :class:`float`, **default=0.975** --
            The y-coordinate for the positioning of text, labelling categories, in the ICSA visualization.

        y_space: :class:`float`, **default=0.05** --
            The y-space between each line of text for the category labels in the ICSA visualization.

    Attributes:
        _metadata

        _model

        _app

        TODO: add other attributes

    """
    def __init__(self,
                 optimisation_direction: Literal['min', 'max'],
                 mode: Literal['concatenate', 'overlay'] = 'concatenate',

                 feature_preprocessing: str = 'unscaled',
                 parameter_preprocessing: str = 'encode and scale',
                 performance_preprocessing: Optional[str] = None,
                 
                 instance_encoder_num_hidden_layers: Optional[int] = None,
                 instance_encoder_num_hidden_units: int = 64,
                 instance_encoder_activation: str = 'relu',

                 instance_decoder_num_hidden_layers: int = 0,
                 instance_decoder_num_hidden_units: int = 64,
                 instance_decoder_activation: str = 'relu',

                 configuration_encoder_num_hidden_layers: int = 3,
                 configuration_encoder_num_hidden_units: int = 64,
                 configuration_encoder_activation: str = 'relu',

                 performance_num_hidden_layers: int = 3,
                 performance_num_hidden_units: int = 64,
                 performance_activation: str = 'relu',

                 optimizer: str = "Adam",
                 performance_loss_weight: float = 10.0,
                 feature_loss_weight: float = 1.0,

                 num_projections: Literal['one', 'two'] = 'two',

                 focus_size: int = 10,
                 unfocus_size: int = 7,
                 instance_colorscale: str = 'Viridis',
                 instance_symbol: str = 'circle',
                 configuration_colorscale: str = 'Plasma',
                 configuration_symbol: str = 'triangle-up',
                 performance_metric_name: str = 'y',

                 min_height: int = 750,
                 min_width: int = 750,
                 text_x: float = 0.025,
                 text_y: float = 0.975,
                 y_space: float = 0.05

                 ):

        # Parameters
        self.mode = mode
        if self.mode not in {'concatenate', 'overlay'}:
            assert False, "The parameter 'mode' must be either 'concatenate' or 'overlay'."
        self.y_space = y_space
        self.text_y = text_y
        self.text_x = text_x
        self.min_width = min_width
        self.min_height = min_height
        self.performance_metric_name = performance_metric_name
        self.configuration_symbol = configuration_symbol
        self.configuration_colorscale = configuration_colorscale
        self.instance_symbol = instance_symbol
        self.instance_colorscale = instance_colorscale
        self.unfocus_size = unfocus_size
        self.focus_size = focus_size
        self.num_projections = num_projections
        self.feature_loss_weight = feature_loss_weight
        self.performance_loss_weight = performance_loss_weight
        self.optimizer = optimizer
        self.performance_activation = performance_activation
        self.performance_num_hidden_units = performance_num_hidden_units
        self.performance_num_hidden_layers = performance_num_hidden_layers
        self.configuration_encoder_activation = configuration_encoder_activation
        self.configuration_encoder_num_hidden_units = configuration_encoder_num_hidden_units
        self.configuration_encoder_num_hidden_layers = configuration_encoder_num_hidden_layers
        self.instance_decoder_activation = instance_decoder_activation
        self.instance_decoder_num_hidden_units = instance_decoder_num_hidden_units
        self.instance_decoder_num_hidden_layers = instance_decoder_num_hidden_layers
        self.instance_encoder_activation = instance_encoder_activation
        self.instance_encoder_num_hidden_units = instance_encoder_num_hidden_units
        # set self.instance_encoder_num_hidden_layers
        if instance_encoder_num_hidden_layers is None:
            self.instance_encoder_num_hidden_layers = 0 if self.mode == 'concatenate' else (
                3 if self.mode == 'overlay' else instance_encoder_num_hidden_layers)
        else:
            self.instance_encoder_num_hidden_layers = instance_encoder_num_hidden_layers
        # set self.performance_preprocessing
        if performance_preprocessing is None:
            self.performance_preprocessing = 'unscaled' if self.mode == 'concatenate' else (
                'minmax scale rows' if self.mode == 'overlay' else performance_preprocessing)
        else:
            self.performance_preprocessing = performance_preprocessing
        self.parameter_preprocessing = parameter_preprocessing
        self.feature_preprocessing = feature_preprocessing
        self.optimisation_direction = optimisation_direction
        self.preprocessor = Preprocessor()

        # Objects

        self.preprocessor = Preprocessor(
            feature_preprocessing=self.feature_preprocessing,
            parameter_preprocessing=self.parameter_preprocessing,
            performance_preprocessing=self.performance_preprocessing,
            optimisation_direction=self.optimisation_direction
        )

        if self.mode == 'concatenate':
            self.shared_space = False
            performance_layer = ConcatenateForward(
                feedforward_network=Feedforward(
                    num_hidden_layers=self.performance_num_hidden_layers,
                    num_hidden_units=self.performance_num_hidden_units,
                    activation=self.performance_activation,
                )
            )
        elif self.mode == 'overlay':
            self.shared_space = True
            performance_layer = OverlayingForward()
        else:
            assert False, "mode == 'concatenate' or 'overlay'"

        self.projector = Projector(
            instance_encoder=Feedforward(
                num_hidden_layers=self.instance_encoder_num_hidden_layers,
                num_hidden_units=self.instance_encoder_num_hidden_units,
                activation=self.instance_encoder_activation
            ),
            instance_decoder=Feedforward(
                num_hidden_layers=self.instance_decoder_num_hidden_layers,
                num_hidden_units=self.instance_decoder_num_hidden_units,
                activation=self.instance_decoder_activation
            ),
            configuration_encoder=Feedforward(
                num_hidden_layers=self.configuration_encoder_num_hidden_layers,
                num_hidden_units=self.configuration_encoder_num_hidden_units,
                activation=self.configuration_encoder_activation
            ),
            configuration_decoder=None,
            performance_model=performance_layer,
            optimizer=self.optimizer,
            loss=Losses(
                performance_loss_weight=self.performance_loss_weight,
                feature_loss_weight=self.feature_loss_weight
            ),
            num_projections=self.num_projections
        )

        self.plotter = Plotter(
            main_plot=MainPlot(
                focus_size=self.focus_size,
                unfocus_size=self.unfocus_size,
                instance_colorscale=self.instance_colorscale,
                instance_symbol=self.instance_symbol,
                configuration_colorscale=self.configuration_colorscale,
                configuration_symbol=self.configuration_symbol,
                shared_space=self.shared_space
            ),
            min_height=self.min_height,
            min_width=self.min_width,
            text_x=self.text_x,
            text_y=self.text_y,
            y_space=self.y_space,
            performance_metric_name=self.performance_metric_name
        )

        self._metadata = None
        self._model = None
        self._app = None

    def fit(self, feature_data: pd.DataFrame, parameter_data: pd.DataFrame, performance_data: pd.DataFrame,
            instance_sources: Optional[pd.Series] = None, configuration_sources: Optional[pd.Series] = None,
            validation_split: float = 0.1, epochs: int = 150, verbose='auto'):

        self._metadata = self.preprocessor.preprocess(feature_data, parameter_data, performance_data)
        performance_matrix = long_to_wide(feature_data, parameter_data, performance_data).to_numpy()

        self._model = self.projector.fit(
            self._metadata.preprocessed_features,
            self._metadata.preprocessed_parameters,
            self._metadata.preprocessed_performance,
            validation_split=validation_split,
            epochs=epochs,
            verbose=verbose
        )

        z = self._model.instance_encoder(self._metadata.preprocessed_features).numpy()
        if self.projector.num_projections == 'two':
            c = self._model.configuration_encoder(self._metadata.preprocessed_parameters).numpy()
        else:
            c = None

        self._feature_data = feature_data
        self._parameter_data = parameter_data
        self._performance_matrix = performance_matrix
        self._z = z
        self._c = c

    def show(self, parameter_design, parameter_domain, parameter_type):

        # FIXME: uses parameter design for parameter data, and it assumes this is given
        # FIXME: instance/configuration sources

        self._app = self.plotter.get(
            self._feature_data, parameter_design, self._performance_matrix,
            self._z, c=self._c,
            # instance_sources=self._instance_sources,
            # configuration_sources=self._configuration_sources,
            parameter_design=parameter_design,
            parameter_domain=parameter_domain,
            parameter_type=parameter_type
        )
        self._app.show()

    # TODO: save the projection models
    def save_fitted_model(self, fpath):
        np.save(fpath + 'z.npy', self._z)
        np.save(fpath + 'c.npy', self._c)
        np.save(fpath + 'performance_matrix.npy', self._performance_matrix)

    def load_fitted_model(self, fpath, feature_data: pd.DataFrame, parameter_data: pd.DataFrame):
        self._z = np.load(fpath + 'z.npy', allow_pickle=True)
        self._c = np.load(fpath + 'c.npy', allow_pickle=True)
        self._performance_matrix = np.load(fpath + 'performance_matrix.npy', allow_pickle=True)
        self._feature_data = feature_data
        self._parameter_data = parameter_data

    def is_fitted(self, fpath: str):
        return os.path.exists(fpath + 'z.npy')

    def fit_show(self, model_fpath: str, metadata: SimpleNamespace, seed: int = 0,
                 validation_split: float = 0.1, epochs: int = 150, save: bool = True,
                 verbose='auto'):
        """
        :param model_fpath: filepath to save and load fitted model
        :param metadata: the output of 'load_metadata' below, with possible case-specific corrections
        :param seed: random seed
        :param verbose: whether to show keras output or not
        :return:
        """
        set_seeds(seed)

        if save:
            if self.is_fitted(model_fpath):
                self.load_fitted_model(model_fpath, metadata.feature_data, metadata.parameter_data)
                self.show(
                    metadata.parameter_design,
                    metadata.parameter_domain,
                    metadata.parameter_type
                )
            else:
                self.fit(metadata.feature_data, metadata.parameter_data, metadata.performance_data,
                         validation_split=validation_split, epochs=epochs, verbose=verbose)
                self.save_fitted_model(model_fpath)
                # load parameter space
                # FIXME: do not require parameter_design
                self.show(
                    metadata.parameter_design,
                    metadata.parameter_domain,
                    metadata.parameter_type
                )
        else:
            self.fit(
                metadata.feature_data,
                metadata.parameter_data,
                metadata.performance_data,
                validation_split=validation_split,
                epochs=epochs,
                verbose=verbose
            )
