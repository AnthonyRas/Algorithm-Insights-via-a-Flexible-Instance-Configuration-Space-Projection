import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import iqr

def median_with_notches(data):
    """
    Calculate the median and its notches with median +- 1.58 * iqr(data) / np.sqrt(n).
    """
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    elif n <= 1:
        return data[0], data[0], data[0]

    median = np.median(data)

    ci_pm = 1.58 * iqr(data) / np.sqrt(n)

    ci_lower, ci_upper = (median - ci_pm, median + ci_pm)


    return median, ci_lower, ci_upper

def compute_grouped_medians(df, values, group_by_columns):
    """
    Compute median_with_notches for each group.
    """
    if len(values) != len(df):
        raise ValueError("Length of 'values' array must match the number of rows in the DataFrame.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("'values' array must contain numeric data.")

    group_by_columns = [col for col in group_by_columns if col is not None]
    temp_df = df.copy()
    temp_df['values'] = values

    if group_by_columns:
        grouped = temp_df.groupby(group_by_columns)['values'].apply(
            lambda x: median_with_notches(x.values)
        ).reset_index()
        grouped[['median', 'ci_lower', 'ci_upper']] = pd.DataFrame(grouped['values'].tolist())
        grouped = grouped.drop(columns=['values'])
    else:
        median, ci_lower, ci_upper = median_with_notches(values)
        grouped = pd.DataFrame({
            'Overall': ['Overall'],
            'median': [median],
            'ci_lower': [ci_lower],
            'ci_upper': [ci_upper]
        })

    return grouped


def plot_grouped_medians(grouped_df, group_by_columns, category_orders=None):
    """
    Plot the grouped median_with_notches using Plotly Express.
    """

    group_by_columns = [col for col in group_by_columns if col is not None]

    grouped_df['error_y'] = grouped_df['ci_upper'] - grouped_df['median']
    grouped_df['error_y_minus'] = grouped_df['median'] - grouped_df['ci_lower']

    params = {
        'data_frame': grouped_df,
        'y': 'median',
        'error_y': 'error_y',
        'error_y_minus': 'error_y_minus',
        'title': 'Median ± 1.58 IQR / SQRT(n)',
        'labels': {'median': 'Median Value'},
        'category_orders': category_orders
    }

    if group_by_columns:
        if len(group_by_columns) >= 1:
            params['x'] = group_by_columns[0]
        if len(group_by_columns) >= 2:
            params['color'] = group_by_columns[1]
        if len(group_by_columns) >= 3:
            params['facet_col'] = group_by_columns[2]
        if len(group_by_columns) > 3:
            print("Warning: More than three grouping columns provided. Additional columns will be ignored in the plot.")
    else:
        params['x'] = 'Overall'

    fig = px.scatter(**params)
    return fig