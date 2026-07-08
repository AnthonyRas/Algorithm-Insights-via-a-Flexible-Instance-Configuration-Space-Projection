import numpy as np
import plotly.graph_objects as go
from typing import Optional, Literal
import pandas as pd
import panel as pn
import param
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.models.widgets.tables import NumberFormatter
from plotly.express.colors import qualitative, sequential
from copy import deepcopy
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from space_updates import CategoricalColours
from types import SimpleNamespace
from space_updates import UpdateCoordinator
from median_plots import *


class MainPlot:
    def __init__(self,
                 focus_size: int = 10,
                 unfocus_size: int = 7,
                 instance_colorscale: str = 'Viridis',
                 instance_symbol: str = 'circle',
                 configuration_colorscale: str = 'Plasma',
                 configuration_symbol: str = 'triangle-up',
                 shared_space: bool = False):
        self.focus_size = focus_size
        self.unfocus_size = unfocus_size
        self.instance_colorscale = instance_colorscale
        self.instance_symbol = instance_symbol
        self.configuration_colorscale = configuration_colorscale
        self.configuration_symbol = configuration_symbol
        self.shared_space = shared_space

    def get(self, z: np.ndarray, instance_values: np.array,
            c: Optional[np.ndarray] = None, configuration_values: Optional[np.array] = None) -> go.Figure:
        n = z.shape[0]
        instance_space_plot = go.Scatter(
            x=z[:, 0], y=z[:, 1], mode='markers',
            marker=dict(
                color=instance_values,
                colorscale=self.instance_colorscale,
                symbol=self.instance_symbol,
                size=self.focus_size,
                colorbar=dict(x=1.15, y=0.25, len=0.4)
            ),
            name='Z', unselected={'marker': {'opacity': 1.0}}
        )
        if c is not None:
            r = c.shape[0]
            configuration_space_plot = go.Scatter(
                x=c[:, 0], y=c[:, 1], mode='markers',
                xaxis='x2', yaxis='y2',
                marker=dict(
                    color=configuration_values,
                    colorscale=self.configuration_colorscale,
                    symbol=self.configuration_symbol,
                    size=self.unfocus_size,
                    colorbar=dict(x=1.15, y=0.8, len=0.4)
                ),
                name='C', unselected={'marker': {'opacity': 1.0}}
            )
            figure = go.Figure(data=[configuration_space_plot, instance_space_plot],
                               layout=go.Layout(font=dict(size=20)))
        else:
            figure = go.Figure(data=[instance_space_plot])
        figure.layout.plot_bgcolor = "rgb(255, 255, 255)"
        figure.layout.paper_bgcolor = "rgb(255, 255, 255)"
        figure.add_shape(
            type='rect',
            xref='paper', yref='paper',
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color='black', width=2),
            fillcolor='rgba(0,0,0,0)'  # transparent fill
        )
        figure.layout.autosize = True
        figure.layout.dragmode = 'lasso'
        zxmin, zxmax = z[:, 0].min(), z[:, 0].max()
        zxrange = zxmax - zxmin
        zymin, zymax = z[:, 1].min(), z[:, 1].max()
        zyrange = zymax - zymin
        if c is not None:
            cxmin, cxmax = c[:, 0].min(), c[:, 0].max()
            cxrange = cxmax - cxmin
            cymin, cymax = c[:, 1].min(), c[:, 1].max()
            cyrange = cymax - cymin
        if self.shared_space:
            xmin = min(c[:, 0].min(), z[:, 0].min()) if c is not None else z[:, 0].min()
            xmax = max(c[:, 0].max(), z[:, 0].max()) if c is not None else z[:, 0].max()
            ymin = min(c[:, 1].min(), z[:, 1].min()) if c is not None else z[:, 1].min()
            ymax = max(c[:, 1].max(), z[:, 1].max()) if c is not None else z[:, 1].max()
            zxmin = xmin
            zxmax = xmax
            zymin = ymin
            zymax = ymax
            cxmin = xmin
            cxmax = xmax
            cymin = ymin
            cymax = ymax
        else:
            zxmin = z[:, 0].min()
            zxmax = z[:, 0].max()
            zymin = z[:, 1].min()
            zymax = z[:, 1].max()
            cxmin = c[:, 0].min()
            cxmax = c[:, 0].max()
            cymin = c[:, 1].min()
            cymax = c[:, 1].max()
        zxrange = zxmax - zxmin
        zyrange = zymax - zymin
        cxrange = cxmax - cxmin
        cyrange = cymax - cymin
        # TODO: DIFFERENT AXIS SCALES FOR CONFIGURATION SPACE IF NOT USING RADIAL
        figure.update_layout(
            template='plotly_white',
            hoverdistance=5, margin=dict(t=40, b=90, pad=10, r=165),
            xaxis=dict(range=[zxmin - 0.05*zxrange, zxmax + 0.05*zxrange], title='Z1', overlaying='free', zeroline=False, showgrid=False),
            yaxis=dict(range=[zymin - 0.05*zyrange, zymax + 0.05*zyrange], title='Z2', title_standoff=0, overlaying='free', zeroline=False, showgrid=False)
        )
        if c is not None:
            figure.update_layout(
                xaxis2=dict(range=[cxmin - 0.05*cxrange, cxmax + 0.05*cxrange], title='C1', overlaying='x', side='top', zeroline=False, showgrid=False),
                yaxis2=dict(range=[cymin - 0.05*cyrange, cymax + 0.05*cyrange], title='C2', title_standoff=0, overlaying='y', side='right', zeroline=False, showgrid=False)
            )
        figure.layout.uirevision = 0
        return figure


def get_variable_type_dict(dataframe: pd.DataFrame):
    variable_type = dict()
    number_vars = dataframe.select_dtypes(include=['number']).columns.to_list()
    cat_vars = dataframe.select_dtypes(exclude=['number']).columns.to_list()
    for v in number_vars:
        variable_type[v] = 'n'
    for v in cat_vars:
        variable_type[v] = 'c'
    return variable_type


class PerformanceVector(param.Parameterized):
    vec = param.Array()

    def __call__(self, array):
        self.vec = array


class Plotter:
    def __init__(self,
                 main_plot=MainPlot(),
                 min_height: int = 750,
                 min_width: int = 750,
                 text_x: float = 0.025,
                 text_y: float = 0.975,
                 y_space: float = 0.05,
                 performance_metric_name: str = 'y'):
        self.main_plot = main_plot
        self.min_height = min_height
        self.min_width = min_width
        self.text_x = text_x
        self.text_y = text_y
        self.y_space = y_space
        self.performance_metric_name = performance_metric_name

    def get(self,
            feature_data: pd.DataFrame,
            parameter_data: pd.DataFrame,
            performance_matrix: np.ndarray,
            z: np.ndarray,
            c: Optional[np.ndarray] = None,
            instance_sources: Optional[pd.Series] = None,
            configuration_sources: Optional[pd.Series] = None,
            instance_variable_type: dict = None,
            configuration_variable_type: dict = None,
            parameter_design: Optional[pd.DataFrame] = None,
            parameter_domain: dict = None,
            parameter_type: dict = None):
        n = feature_data.shape[0]
        # TODO: also have an option to replace below (both instance and configuration) with input to Plotter
        # TODO: code assumes parameter_domain and parameter_type are given
        if instance_variable_type is None:
            instance_variable_type = get_variable_type_dict(feature_data)
        r = parameter_data.shape[0]
        if configuration_variable_type is None:
            configuration_variable_type = get_variable_type_dict(parameter_data)
        instances_selected = np.ones(n, dtype=bool)
        instance_aggregate_performances = performance_matrix.mean(axis=1)
        configurations_selected = np.ones(r, dtype=bool)
        configuration_aggregate_performances = performance_matrix.mean(axis=0)
        # Instance/Configuration Space Figure
        figure = self.main_plot.get(
            z, instance_aggregate_performances,
            c=c, configuration_values=(configuration_aggregate_performances if c is not None else None)
        )
        plotly_pane = pn.pane.Plotly(
            figure, min_height=self.min_height, min_width=self.min_width,
            config={'responsive': True, 'doubleClick': 'reset', 'modeBarButtonsToRemove': ['autoScale2d'],
                    'toImageButtonOptions': {'format': 'png', 'scale': 2}},
        )

        if configuration_sources is None:
            configuration_plot_options = ['Aggregate Performance'] + list(parameter_data.columns)
            configuration_sources = pd.Series(["S" for i in range(parameter_data.shape[0])]).astype('str')
            csource_levels = configuration_sources.drop_duplicates(keep='first').values.flatten().tolist()
            csource_levels.sort(reverse = True)
            csource_levels, csource_level_colours, csource_levels_to_colours, csource_colour_vector = CategoricalColours(
                csource_levels, 'c').get_categorical_colours(configuration_sources)
        else:
            configuration_plot_options = ['Aggregate Performance'] + ['Configuration Cluster'] + list(
                parameter_data.columns)
            csource_levels = configuration_sources.drop_duplicates(keep='first').values.flatten().tolist()
            csource_levels.sort(reverse = True)
            csource_levels, csource_level_colours, csource_levels_to_colours, csource_colour_vector = CategoricalColours(
                csource_levels, 'c').get_categorical_colours(configuration_sources)

        if instance_sources is None:
            instance_plot_options = ['Aggregate Performance'] + list(feature_data.columns)
            instance_sources = pd.Series(["S" for i in range(feature_data.shape[0])]).astype('str')
            isource_levels = instance_sources.drop_duplicates(keep='first').values.flatten().tolist()
            isource_levels.sort(reverse = True)
            isource_levels, isource_level_colours, isource_levels_to_colours, isource_colour_vector = CategoricalColours(
                isource_levels, 'c').get_categorical_colours(instance_sources)
        else:
            # TODO: do for configurations too
            # FIXME: ASSUMES configuration_sources IS NOT NONE
            instance_plot_options = ['Aggregate Performance'] + ['Best Configuration Cluster (Median)'] + list(feature_data.columns)
            isource_levels = instance_sources.drop_duplicates(keep='first').values.flatten().tolist()
            isource_levels.sort(reverse = True)

            isource_levels, isource_level_colours, isource_levels_to_colours, isource_colour_vector = CategoricalColours(
                isource_levels, 'c').get_categorical_colours(instance_sources)

        select_feature = pn.widgets.Select(
            name='Plot (Instance Space)', value='Aggregate Performance',
            options=instance_plot_options
        )
        reset_instances_button = pn.widgets.Button(
            name='Instances', width=120, button_style='outline', button_type='light'
        )

        colour_select = pn.widgets.Select(name='Colour', options=[None]+list(parameter_design.columns), height=50)
        xaxis_select = pn.widgets.Select(name='X-Axis', options=[None]+list(parameter_design.columns), height=50)
        facet_select = pn.widgets.Select(name='Facet', options=[None]+list(parameter_design.columns), height=50)

        parameter_data_with_performance = parameter_data.copy()
        parameter_data_with_performance.insert(0,
            'aggregate_performance', configuration_aggregate_performances)

        all_configurations = pn.widgets.Tabulator(
            parameter_data_with_performance, sizing_mode='stretch_both', name='Selected Configurations', text_align='center', theme='midnight',
            editors=dict(
                zip(parameter_data_with_performance.columns.to_list(), [None for _ in range(len(parameter_data_with_performance.columns.to_list()))])
            )
        )

        qt = QuantileTransformer()
        qfeatures = qt.fit_transform(feature_data)
        qfeatures = pd.DataFrame(qfeatures, columns=qt.feature_names_in_, index=feature_data.index)
        column_widths = {feature: 60 for feature in qfeatures.columns}
        column_widths['Row'] = 60

        num_formatter = NumberFormatter(format='0.00%')

        all_instances = pn.widgets.Tabulator(
            qfeatures, sizing_mode='stretch_both', name='Features of Selected Instances (Percentile Ranks)',
            editors=dict(
                zip(qfeatures.columns.to_list(), [None for _ in range(len(qfeatures.columns.to_list()))])
            )
            , widths=column_widths, formatters={feature: num_formatter for feature in qfeatures.columns}, theme='midnight'
        )

        def map_colours(val):
            colour = mcolors.to_hex(plt.cm.cividis(val))
            return 'background-color: %s' % colour

        all_instances.style.map(map_colours)

        # FIXME: USE THE SAME FOR ALL FIGURES? OR USE SELECTED CONFIGS / INSTANCES?
        yvec = PerformanceVector(vec=configuration_aggregate_performances)

        @pn.depends(xaxis_select.param.value, colour_select.param.value, facet_select.param.value, yvec.param.vec)
        def update_boxplot(x, colour, facet, y):
            category_orders = {parameter: [str(lev) for lev in parameter_domain[parameter]] for parameter in parameter_domain.keys()}

            fig = px.box(
                parameter_design, x=x, y=y, color=colour, facet_col=facet, points='all',
                category_orders=category_orders
            ).update_layout(
                template='plotly_white', autosize=True, font=dict(size=20)).update_traces(
                marker=dict(symbol='circle-open', size=12), pointpos=0.0, jitter=0.0
            ).update_yaxes(title_text='Aggregate ' + self.performance_metric_name + ' over selected instances')

            return fig

        boxplot_pane = pn.pane.Plotly(
            update_boxplot, config={'toImageButtonOptions': {'format': 'png', 'scale': 2}}
        )

        @pn.depends(xaxis_select.param.value, colour_select.param.value, facet_select.param.value, yvec.param.vec)
        def update_notches(x, colour, facet, y):
            category_orders = {parameter: [str(lev) for lev in parameter_domain[parameter]] for parameter in parameter_domain.keys()}

            group_by_columns = [x, colour, facet]
            grouped = compute_grouped_medians(parameter_design, y, group_by_columns)
            fig = plot_grouped_medians(grouped, group_by_columns, category_orders=category_orders).update_layout(
                template='plotly_white', autosize=True, font=dict(size=20)).update_yaxes(title_text='Aggregate ' + self.performance_metric_name + ' over selected instances')

            return fig

        notch_pane = pn.pane.Plotly(
            update_notches, config={'toImageButtonOptions': {'format': 'png', 'scale': 2}}
        )

        if c is not None:
            select_parameter = pn.widgets.Select(
                name='Plot (Configuration Space)', value='Aggregate Performance',
                options=configuration_plot_options
            )
            choose_focus = pn.widgets.RadioButtonGroup(
                name='Focus', options=['Instance Space', 'Configuration Space'],
                value='Instance Space', button_style='outline', button_type='light'
            )
            reset_configurations_button = pn.widgets.Button(
                name='Configurations', width=120, button_style='outline', button_type='light'
            )

        cn = SimpleNamespace(
            feature_data=feature_data,
            parameter_data=parameter_data,
            performance_matrix=performance_matrix,
            z=z,
            c=c,
            instance_sources=instance_sources,
            configuration_sources=configuration_sources,
            instance_variable_type=instance_variable_type,
            configuration_variable_type=configuration_variable_type,
            parameter_design=parameter_design,
            parameter_domain=parameter_domain,
            parameter_type=parameter_type,
            main_plot=self.main_plot,
            yvec=yvec,
            configuration_aggregate_performances=configuration_aggregate_performances,
            parameter_data_with_performance=parameter_data_with_performance,
            all_configurations=all_configurations,
            configurations_selected=configurations_selected,
            all_instances=all_instances,
            qfeatures=qfeatures,
            instances_selected=instances_selected,
            plotly_pane=plotly_pane,
            choose_focus=choose_focus,
            n=n,
            r=r,
            instance_aggregate_performances=instance_aggregate_performances,
            select_feature=select_feature,
            select_parameter=select_parameter,
            isource_colour_vector=isource_colour_vector,
            isource_levels=isource_levels,
            isource_level_colours=isource_level_colours,
            csource_colour_vector=csource_colour_vector,
            csource_levels=csource_levels,
            csource_level_colours=csource_level_colours,
            performance_metric_name=self.performance_metric_name
        )

        uc = UpdateCoordinator(cn)

        select_feature.param.watch(uc.com.instance_space.update_feature, 'value')
        plotly_pane.param.watch(uc.com.on_select, 'selected_data')
        reset_instances_button.param.watch(uc.com.instance_space.reset_instances, 'value')

        if c is not None:
            select_parameter.param.watch(uc.com.configuration_space.update_parameter, 'value')
            choose_focus.param.watch(uc.com.change_focus, 'value')
            reset_configurations_button.param.watch(uc.com.configuration_space.reset_configurations, 'value')
        # app
        if c is not None:
            app = pn.Row(
                pn.WidgetBox(pn.Row(pn.Row(
                    pn.Column(select_feature, select_parameter), pn.Row(pn.layout.HSpacer(), pn.Column("### Focus:", choose_focus), pn.layout.HSpacer(), pn.Column("### Reset Selection:",
                    pn.Row(reset_instances_button, reset_configurations_button)), pn.layout.HSpacer())
                    )), pn.layout.Spacer(height=25),
                    pn.Tabs(pn.WidgetBox(pn.Row(xaxis_select, colour_select, facet_select), notch_pane, name='Notches'),
                            pn.WidgetBox(pn.Row(xaxis_select, colour_select, facet_select), boxplot_pane, name='Boxplots'), all_configurations, all_instances, tabs_location='below')
                ), plotly_pane
            )
        else:
            app = pn.Row(
                pn.WidgetBox(pn.Row(pn.Column(
                    select_feature, "### Reset Selection:",
                    pn.Row(reset_instances_button)
                )),
                    pn.Tabs(pn.WidgetBox(pn.Row(xaxis_select, colour_select, facet_select), boxplot_pane, name='Boxplots'), tabs_location='below')
                ), plotly_pane
            )
        return app
