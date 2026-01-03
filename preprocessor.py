from typing import Literal, Optional
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, MinMaxScaler


def encode_and_scale(data: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    categorical_columns = list(data.select_dtypes(include=['object', 'category']).columns)
    numeric_columns = list(filter(lambda x: x not in categorical_columns, list(data.columns)))
    data_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(
                steps=[("scaler", StandardScaler())]
            ), numeric_columns),
            ("categorical", Pipeline(
                steps=[("encoder", OneHotEncoder(sparse_output=False)), ("scaler", StandardScaler())]
            ), categorical_columns)
        ]
    )
    data_matrix = data_preprocessor.fit_transform(data)
    return data_matrix, data_preprocessor


class PreprocessorUnscaled:
    def __init__(self):
        ...

    def fit(self):
        ...

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        return data.to_numpy()

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return data.to_numpy()


def preprocess_attributes(data: pd.DataFrame, option: str):
    if option == 'encode and scale':
        return encode_and_scale(data)
    elif option == 'unscaled':
        preproc = PreprocessorUnscaled()
        data = preproc.fit_transform(data)
        return data, preproc
    else:
        assert False, 'Please select one of the available preprocessing options.'


def transpose_transform(d: np.ndarray) -> np.ndarray:
    return d.transpose()


def negative_transform(d: np.ndarray) -> np.ndarray:
    return -d


class PerformancePreprocessorMinmaxRows:
    def __init__(self, optimisation_direction):
        self._optimisation_direction = optimisation_direction
        self._transposer = None
        self._reflector = None
        self._minmax = None
        self._data_preprocessor = None

    def fit(self, performance_matrix: np.ndarray):
        self._transposer = FunctionTransformer(func=transpose_transform, inverse_func=transpose_transform)
        self._reflector = FunctionTransformer(func=negative_transform, inverse_func=negative_transform)
        self._minmax = MinMaxScaler()
        if self._optimisation_direction == 'min':
            self._data_preprocessor = Pipeline(
                steps=[("reflect", self._reflector), ("t_in", self._transposer), ("minmax", self._minmax), ("t_out", self._transposer)]
            )
        else:
            assert self._optimisation_direction == 'max', 'optimisation_direction must be min or max'
            self._data_preprocessor = Pipeline(
                steps=[("t_in", self._transposer), ("minmax", self._minmax), ("t_out", self._transposer)]
            )
        self._data_preprocessor.fit(performance_matrix)

    def transform(self, performance_matrix: np.ndarray) -> np.ndarray:
        return self._data_preprocessor.transform(performance_matrix)

    def fit_transform(self, performance_matrix: np.ndarray) -> np.ndarray:
        self.fit(performance_matrix)
        return self.transform(performance_matrix)


class PerformancePreprocessorUnscaled:
    def __init__(self):
        ...

    def fit(self):
        ...

    def transform(self, performance_matrix: np.ndarray) -> np.ndarray:
        return performance_matrix

    def fit_transform(self, performance_matrix: np.ndarray) -> np.ndarray:
        return performance_matrix


# convert long format performance (instance ID, configuration ID, value) to wide format
# TODO: assumes every instance and configuration are evaluated at least once
def long_to_wide(feature_data: pd.DataFrame, parameter_data: pd.DataFrame, performance_data: pd.DataFrame) -> pd.DataFrame:
    performance_matrix = performance_data.pivot(
        index=performance_data.columns[0], columns=performance_data.columns[1], values=performance_data.columns[2]
    )
    return performance_matrix[parameter_data.index].loc[feature_data.index]

def preprocess_performance(
        feature_data: pd.DataFrame, parameter_data: pd.DataFrame, performance_data: pd.DataFrame, option: str
):
    performance_matrix = long_to_wide(feature_data, parameter_data, performance_data)

    assert len(list(performance_matrix.select_dtypes(include=['object', 'category']).columns)) == 0,\
        "DataFrame must be numeric."

    if option == 'minmax scale rows: minimisation':
        preprocessor = PerformancePreprocessorMinmaxRows('min')
    elif option == 'minmax scale rows: maximisation':
        preprocessor = PerformancePreprocessorMinmaxRows('max')
    elif option == 'unscaled':
        preprocessor = PerformancePreprocessorUnscaled()
    else:
        assert False, 'Please select one of the available preprocessing options.'

    data_matrix = preprocessor.fit_transform(performance_matrix.to_numpy())
    data_matrix = pd.DataFrame(data_matrix, index=performance_matrix.index, columns=performance_matrix.columns)
    preprocessed_performance = data_matrix.melt(
        ignore_index=False, var_name=performance_data.columns[1], value_name=performance_data.columns[2]
    ).reset_index(names=performance_data.columns[0])
    preprocessed_performance.iloc[:, 0] = preprocessed_performance.iloc[:, 0].map(
        dict(zip(feature_data.index, range(len(feature_data.index)))))
    preprocessed_performance.iloc[:, 1] = preprocessed_performance.iloc[:, 1].map(
        dict(zip(parameter_data.index, range(len(parameter_data.index)))))

    return preprocessed_performance, preprocessor


class Preprocessor:
    def __init__(self,
                 feature_preprocessing: str = 'unscaled',
                 parameter_preprocessing: str = 'encode and scale',
                 performance_preprocessing: str = 'unscaled',
                 optimisation_direction: Optional[Literal['min', 'max']] = None):
        self.feature_preprocessing = feature_preprocessing
        self.parameter_preprocessing = parameter_preprocessing
        self.performance_preprocessing = performance_preprocessing
        self.optimisation_direction = optimisation_direction

    def preprocess(self, feature_data: pd.DataFrame, parameter_data: pd.DataFrame, performance_data: pd.DataFrame):
        preprocessed_features, feature_preprocessor = preprocess_attributes(
            feature_data, self.feature_preprocessing)
        preprocessed_parameters, parameter_preprocessor = preprocess_attributes(
            parameter_data, self.parameter_preprocessing)

        if self.performance_preprocessing == 'minmax scale rows':
            if self.optimisation_direction == 'min':
                self.performance_preprocessing = 'minmax scale rows: minimisation'
            elif self.optimisation_direction == 'max':
                self.performance_preprocessing = 'minmax scale rows: maximisation'
            else:
                assert False, "optimisation_direction == 'min' or 'max'"
        preprocessed_performance, performance_preprocessor = preprocess_performance(
            feature_data, parameter_data, performance_data, self.performance_preprocessing)
        # TODO: express preprocessed performance in terms of integer indexes not instance/configuration names
        # TODO: assumes that preprocess_performance returns long format with instance/configuration names
        # TODO: assumes instances in rows, configurations in columns (with row names and column names)
        # TODO: perhaps it should be assumed that the performance data is in 'edgelist format' (convert matrix to this)
        # input for model TODO (perhaps use generators to save space if needed)
        preprocessed_performance = preprocessed_performance.sample(frac=1).reset_index(drop=True)
        return SimpleNamespace(
            feature_data=feature_data.copy(),
            parameter_data=parameter_data.copy(),
            performance_data=performance_data.copy(),
            preprocessed_features=preprocessed_features,
            preprocessed_parameters=preprocessed_parameters,
            preprocessed_performance=preprocessed_performance,
            feature_preprocessor=feature_preprocessor,
            parameter_preprocessor=parameter_preprocessor,
            performance_preprocessor=performance_preprocessor
        )
