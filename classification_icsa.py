import pickle
from types import SimpleNamespace
from typing import Optional, Literal
import pandas as pd
from sklearn import datasets
import re
import glob
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import arff
import json
import numpy as np
from sklearn.metrics._classification import confusion_matrix
from experimental_design import ParameterSpace, NA
import multiprocessing as mp
from functools import partial
import multiprocess as mp
import psutil
import inspect
from icsa import set_seeds

# CREDIT: https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/metrics/_classification.py#L910
def MCC(C):
    """
    :param C: Confusion matrix of shape (num_classes, num_classes), where num_classes is the number of classes.
    :return: Matthews correlation coefficient (MCC), a classification performance measure that takes values between
     -1 to +1, where +1 indicates a perfect prediction, 0 indicates a random prediction, and -1 indicates a perfectly
      wrong prediction.
    """
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


# the imputation process from https://doi.org/10.1007/s10994-017-5629-5
def impute_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    The imputation process from https://doi.org/10.1007/s10994-017-5629-5.

    :param df: The DataFrame to impute missing values.
    :return: The DataFrame with imputed values.
    """
    df = df[~df['class'].isna()]  # use only rows where label is not NA
    for cl in df['class'].cat.categories:
        df_class_subset = df[df['class'] == cl]  # get subset of rows with class 'cl'
        class_dict = dict()  # mean/mode for each numeric/categorical attribute within class 'cl'
        for attribute in df.columns:
            if attribute != 'class':
                is_categorical = isinstance(df[attribute].dtype, pd.CategoricalDtype)
                is_all_na = df_class_subset[attribute].isna().all()
                if is_categorical:
                    if is_all_na:
                        class_dict[attribute] = df[attribute].mode()
                    else:
                        class_dict[attribute] = df_class_subset[attribute].mode()
                else:
                    if is_all_na:
                        class_dict[attribute] = df[attribute].mean()
                    else:
                        class_dict[attribute] = df_class_subset[attribute].mean()
        cl_indexes = df_class_subset.index
        df.loc[cl_indexes] = df_class_subset.fillna(class_dict)
    return df



def is_instance_done(instance_series_file_path: str, instance_name: str):
    if os.path.exists(instance_series_file_path):
        instance_series = pd.read_csv(instance_series_file_path, header=None)[0]
        instance_done = instance_series.isin([instance_name]).any()
    else:
        instance_done = False
    return instance_done

def load_instance(instances_directory: str, instance_name: str):
    # NOTE: assumes the instances are given as .arff files
    with open(instances_directory + instance_name + '.arff', 'r') as file:
        data = arff.load(file)
    df = pd.DataFrame(
        data['data'], columns=[attribute_name for attribute_name, attribute_type in data['attributes']]
    )
    for attribute_name, attribute_type in data['attributes']:
        if isinstance(attribute_type, list):
            df[attribute_name] = df[attribute_name].astype('category')

    df = impute_na(df)  # impute missing values
    X = pd.get_dummies(df.drop('class', axis=1)).astype('float')
    # centre and scale for comparability between classifiers
    scaler = StandardScaler()
    X.loc[:] = scaler.fit_transform(X)
    y = df['class']
    y = pd.Series(y.cat.codes.to_numpy(), dtype='category')
    return X, y


def cv_classifier(clf, X, y, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=None)
    C = confusion_matrix(y, y_pred).tolist()
    return C


def get_instance_names(instances_directory: str):
    """
    :param instances_directory: the directory of the instances in *.arff format
    :return:
    """
    # get the names of instances
    instance_filenames = glob.glob(os.path.join(instances_directory, '*.arff'))
    instance_names = [os.path.basename(x).split('.')[0] for x in instance_filenames]
    instance_names.sort()
    instance_names = list(filter(lambda x: x is not None, instance_names))
    return instance_names

# done for individual classifiers, not sets of parameter configurations
def cv_classifier_all(clf, instances_directory: str, results_path_plus_name: str, n_splits: int = 5, seed: int = 0):
    instance_names = get_instance_names(instances_directory)

    # load existing results, if any
    if os.path.exists(results_path_plus_name):
        with open(results_path_plus_name, 'rb') as file:
            results = pickle.load(file)  # list of: list of instances done and list of confusion matrices
    else:
        results = [[], []]

    for instance_name in tqdm(instance_names):
        set_seeds(seed)
        instances_done = results[0]
        if instance_name in instances_done:
            continue
        X, y = load_instance(instances_directory, instance_name)
        C = cv_classifier(clf, X, y, n_splits=n_splits)
        results[0].append(instance_name)
        results[1].append(C)

        with open(results_path_plus_name, 'wb') as file:
            pickle.dump(results, file)

    return results


def run_classifier(Classifier, value_error_correction_function, X, y, cv, seed, multicore_mode, n_jobs, instance_id, config_idx, params):
    """
    used in 'generate' method below to get classifier performance data

    :param Classifier: sklearn classifier
    :param X: DataFrame
    :param y: class labels
    :param cv: sklearn cross-validation object
    :param seed: random seed
    :param config_idx: index of parameter configuration
    :param params: parameter configuration
    :return: instance_id, config_idx, confusion matrix
    """
    params = {k: (None if isinstance(v, NA) else v) for k, v in params.items()}  # replace NA by None
    if multicore_mode == 'folds':
        # FIXME: FIXME run cross_val_predict with multicore over folds
        n_jobs_active = -1
    else:
        n_jobs_active = None
    if multicore_mode == 'classifier':
        n_jobs_classifier = n_jobs
    else:
        n_jobs_classifier = None
    if 'random_state' in inspect.signature(Classifier).parameters:
        if 'n_jobs' in inspect.signature(Classifier).parameters:
            clf = Classifier(**params, random_state=seed, n_jobs=n_jobs_classifier)
        else:
            clf = Classifier(**params, random_state=seed)
    else:
        if 'n_jobs' in inspect.signature(Classifier).parameters:
            clf = Classifier(**params, n_jobs=n_jobs_classifier)
        else:
            clf = Classifier(**params)
    try:
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=n_jobs_active)
    except ValueError as value_error:
        if 'random_state' in inspect.signature(Classifier).parameters:
            if 'n_jobs' in inspect.signature(Classifier).parameters:
                clf = Classifier(**params, random_state=seed, n_jobs=n_jobs_classifier)
            else:
                clf = Classifier(**params, random_state=seed)
        else:
            if 'n_jobs' in inspect.signature(Classifier).parameters:
                clf = Classifier(**params, n_jobs=n_jobs_classifier)
            else:
                clf = Classifier(**params)
        value_error_correction_function(clf, value_error)
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=n_jobs_active)
    C = confusion_matrix(y, y_pred).tolist()
    result = {'instances': instance_id, 'configurations': config_idx, 'value': C}
    return result


def get_confusion(fpath: str):
    """
    get confusion matrices
    :param fpath:
    :return: list of confusion matrices
    """
    # load instance series
    instance_series_file_path = fpath + 'instance_series.csv'
    instance_series = pd.read_csv(instance_series_file_path, header=None)[0]

    # get confusion matrices for each instance
    performance_data_file_path = fpath
    confusion_matrices = []
    for instance_name in instance_series:
        with open(performance_data_file_path + instance_name + '.json', 'r') as f:
            instance_confusion = json.load(f)
        confusion_matrices.extend(instance_confusion)
    return confusion_matrices


def get_performance_data(fpath: str, confusion_function=MCC):
    """
    from confusion matrices, compute performance metric
    :param fpath: path to main directory containing output files
    :param confusion_function: a classification performance metric to be computed over confusion matrices
    :return:
    """
    confusion_matrices = get_confusion(fpath)
    for idx, d in enumerate(confusion_matrices):
        confusion = np.array(confusion_matrices[idx]['value'])
        confusion_matrices[idx]['value'] = confusion_function(confusion)

    performance_data = pd.DataFrame(confusion_matrices)
    return performance_data


class ClassificationMetadataGenerator:
    def __init__(self, Classifier = DecisionTreeClassifier, n_splits: int = 5,
                 n_jobs: int = -1, value_error_correction_function = lambda c, v: v,
                 multicore_mode: Literal['configs', 'folds', 'classifier'] = 'configs'):
        """

        :param Classifier:
        :param n_splits:
        :param n_jobs:
        :param value_error_correction_function: a function of (classifier, value_error) which takes an initialised
        classifier and a value_error from running 'cross_val_predict' and corrects the parameter values of 'classifier'
        """
        self.Classifier = Classifier
        self.n_splits = n_splits
        if n_jobs == -1 and multicore_mode == 'configs':
            self.n_jobs = psutil.cpu_count(logical=False)//4  # avoid taking up too much memory
        else:
            self.n_jobs = n_jobs
        self.value_error_correction_function = value_error_correction_function
        self.multicore_mode = multicore_mode  # parallel processing over configurations or folds of CV?

    def generate(self, X, y, instance_id: str, parameter_data, seed: int = 0):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=False)  # TODO: stratified?
        results = []
        if self.multicore_mode == 'configs':
            with mp.Pool(processes=self.n_jobs) as pool:
                worker_func = partial(
                    run_classifier, self.Classifier, self.value_error_correction_function, X, y, cv, seed, self.multicore_mode, self.n_jobs, instance_id
                )
                results = pool.starmap(worker_func, enumerate(parameter_data))
                pool.close()
                pool.join()
        elif self.multicore_mode == 'folds' or self.multicore_mode == 'classifier':
            for config_idx, params in enumerate(parameter_data):
                res = run_classifier(self.Classifier, self.value_error_correction_function, X, y, cv, seed, self.multicore_mode, self.n_jobs, instance_id, config_idx, params)
                results.append(res)
        else:
            assert False, "self.multicore_mode == 'configs' or 'folds' or 'classifier'"
        return results

    def set_instance_done(self, fpath: str, instance_name: str):
        instance_series = pd.Series([instance_name])
        instance_series.to_csv(fpath + 'instance_series.csv',
                               mode='a' if os.path.exists(fpath + 'instance_series.csv') else 'w', header=False,
                               index=False)

    def run_configurations(self, instances_directory: str, parameter_space: ParameterSpace):
        instance_names = get_instance_names(instances_directory)  # get the names of classification instances
        fpath = parameter_space.filepath  # get the main filepath where the results are stored
        for instance_name in tqdm(instance_names):
            if is_instance_done(fpath + 'instance_series.csv', instance_name):
                continue
            X, y = load_instance(instances_directory, instance_name)
            performance_data = self.generate(X, y, instance_name, parameter_space.parameter_data, seed=0)
            self.set_instance_done(fpath, instance_name)
            with open(fpath + instance_name + '.json', 'w') as f:
                json.dump(performance_data, f)

    def get_features(self, feature_data_filename: str, fpath: str, instance_names: list[str]):
        feature_data = pd.read_csv(feature_data_filename).set_index('Row')  # FIXME: assumes instance names are given as 'row' column
        feature_data.loc[instance_names].to_csv(fpath + 'feature_data.csv')
        return feature_data

    def run(self, instances_directory: str, feature_data_filename: str, fpath: str, parameter_space: ParameterSpace):
        self.run_configurations(instances_directory, parameter_space)
        instance_names = get_instance_names(instances_directory)
        feature_data = self.get_features(feature_data_filename, fpath, instance_names)


def load_metadata(fpath: str, confusion_function=MCC):
    # features
    feature_data = pd.read_csv(fpath + 'feature_data.csv', index_col=0)
    # parameters
    with open(fpath + 'parameter_data.pkl', 'rb') as file:
        parameter_data = pickle.load(file)
    parameter_data = pd.DataFrame(parameter_data)

    # parameter experimental design
    with open(fpath + 'parameter_design.pkl', 'rb') as file:
        parameter_design = pickle.load(file)
    with open(fpath + 'parameter_domain.pkl', 'rb') as file:
        parameter_domain = pickle.load(file)
    with open(fpath + 'parameter_type.pkl', 'rb') as file:
        parameter_type = pickle.load(file)
    with open(fpath + 'parameter_default.pkl', 'rb') as file:
        parameter_default = pickle.load(file)
    # performance
    performance_data = get_performance_data(
        fpath, confusion_function=confusion_function
    )
    return SimpleNamespace(
        feature_data=feature_data,
        parameter_data=parameter_data,
        parameter_design=parameter_design,
        parameter_domain=parameter_domain,
        performance_data=performance_data,
        parameter_type=parameter_type,
        parameter_default=parameter_default
    )


# functions for loading the metadata from particular classifiers


def load_knn_metadata(knn_fpath: str, confusion_function=MCC):
    knn_metadata = load_metadata(knn_fpath, confusion_function=confusion_function)

    # case-specific data preparation for plotting
    knn_metadata.parameter_design = pd.DataFrame(knn_metadata.parameter_design)
    knn_metadata.parameter_design = knn_metadata.parameter_design.astype('str')

    non_minkowski_idxs = knn_metadata.parameter_design[knn_metadata.parameter_design['metric'] != 'minkowski'].index
    knn_metadata.parameter_design.loc[non_minkowski_idxs, 'p'] = 'n/a'
    knn_metadata.parameter_data.loc[non_minkowski_idxs, 'p'] = 'n/a'
    # knn_metadata.parameter_domain['p'].append('n/a')
    knn_metadata.parameter_type['p'] = 'c'
    knn_metadata.parameter_data['p'] = knn_metadata.parameter_data['p'].astype('str')
    return knn_metadata


def load_sgd_metadata(sgd_fpath: str, confusion_function=MCC):
    sgd_metadata = load_metadata(sgd_fpath, confusion_function=confusion_function)

    # case-specific data preparation for plotting
    sgd_metadata.parameter_design = pd.DataFrame(sgd_metadata.parameter_design)
    sgd_metadata.parameter_design = sgd_metadata.parameter_design.astype('str')

    for parameter in sgd_metadata.parameter_default.keys():
        na_indexes = sgd_metadata.parameter_data[parameter].map(lambda x: isinstance(x, NA))
        if sgd_metadata.parameter_type[parameter] == 'r':
            sgd_metadata.parameter_data.loc[na_indexes, parameter] = sgd_metadata.parameter_default[parameter]

    return sgd_metadata


# functions for convenience


def get_fpath(folder_name: str):
    fpath = 'analysis/classification/' + folder_name + '/'
    return fpath


def get_param_space(folder_name: str, parameter_domain: dict, parameter_type: dict, parameter_default: dict,
                    seed: int = 0):
    set_seeds(seed)
    # get the parameter_space object and filepath
    fpath = get_fpath(folder_name)
    ps = ParameterSpace(fpath, parameter_domain, parameter_type, parameter_default)
    print(ps.to_acts())
    return ps


def get_param_configs(t: int, ps: ParameterSpace, use_log_uniform: list, seed: int = 0):
    set_seeds(seed)
    fpath = ps.filepath
    # get the parameter_space object and filepath
    ps.generate_parameter_configurations(
        fpath + 'parameter_space-t' + str(t) + '.csv', unfilled='*', use_log_uniform=use_log_uniform
    )
    with open(fpath + 'ps.pkl', 'wb') as file:
        pickle.dump(ps, file)


def run_classifier_configs(Classifier, value_error_correction, folder_name: str, seed: int = 0,
                           multicore_mode: Literal['configs', 'folds', 'classifier'] = 'configs'):
    # get performance data
    set_seeds(seed)
    fpath = get_fpath(folder_name)
    with open(fpath + 'ps.pkl', 'rb') as file:
        ps = pickle.load(file)
    cmg = ClassificationMetadataGenerator(
        Classifier=Classifier, value_error_correction_function=value_error_correction, multicore_mode=multicore_mode
    )
    instances_directory = 'analysis/classification/instances/UCI-arff-final/'
    feature_data_filename = 'analysis/classification/features/feature_process.csv'
    fpath = ps.filepath
    cmg.run(instances_directory, feature_data_filename, fpath, ps)
