"""Update functions for the main 2D space plots."""

import panel as pn
import pandas as pd
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import qualitative, sequential


class UpdateCoordinator:
    def __init__(self, cn: SimpleNamespace):
        self.cn = cn  # context: metadata, etc.
        self.com = CombinedSpace(self, self.cn)

    def update_all(self):
        """Method to update on lasso selection changes."""
        self.com.update_on_selection_change()

        self.cn.yvec(self.cn.configuration_aggregate_performances)

        self.cn.parameter_data_with_performance['aggregate_performance'] = self.cn.configuration_aggregate_performances
        self.cn.all_configurations.value = self.cn.parameter_data_with_performance.iloc[self.cn.configurations_selected]
        self.cn.all_instances.value = self.cn.qfeatures.iloc[self.cn.instances_selected]


class CombinedSpace:
    def __init__(
            self,
            uc: UpdateCoordinator,
            cn: SimpleNamespace,
            text_x: float = 0.025,
            text_y: float = 0.975,
            y_space: float = 0.05,
    ):
        self.uc = uc
        self.cn = cn
        self.text_x = text_x
        self.text_y = text_y
        self.y_space = y_space
        #
        self.instance_space = InstanceSpace(self, self.cn)
        self.configuration_space = ConfigurationSpace(self, self.cn)

    def change_focus(self, event):
        trace_names = [trace.name for trace in self.cn.plotly_pane.object.data]
        instance_curve_number = trace_names.index('Z')
        configuration_curve_number = trace_names.index('C')
        with self.cn.plotly_pane.object.batch_update():
            if event.new == 'Instance Space':
                self.cn.plotly_pane.object.data[instance_curve_number].marker.size = 10
                self.cn.plotly_pane.object.data[configuration_curve_number].marker.size = 7
            elif event.new == 'Configuration Space':
                self.cn.plotly_pane.object.data[instance_curve_number].marker.size = 7
                self.cn.plotly_pane.object.data[configuration_curve_number].marker.size = 10

    def on_select(self, event):
        if event.new is not None:
            select_callback_data = event.new['points']
            trace_names = [trace.name for trace in self.cn.plotly_pane.object.data]
            instance_curve_number = trace_names.index('Z')
            if self.cn.c is not None:
                configuration_curve_number = trace_names.index('C')
                select_trace = instance_curve_number if self.cn.choose_focus.value == 'Instance Space' else configuration_curve_number
            else:
                select_trace = instance_curve_number
                configuration_curve_number = None
            selected_point_numbers = [x['pointIndex'] for x in select_callback_data if x['curveNumber'] == select_trace]
            if len(selected_point_numbers) > 0:
                num_points = len(self.cn.plotly_pane.object.data[select_trace].x)
                if instance_curve_number == select_trace:
                    self.cn.instances_selected[:] = np.zeros(self.cn.n, dtype=bool)
                    self.cn.instances_selected[selected_point_numbers] = True
                    self.cn.configuration_aggregate_performances[:] = self.cn.performance_matrix.mean(
                        axis=0, where=self.cn.instances_selected.reshape(-1, 1))
                    symbols = [self.cn.main_plot.instance_symbol + "-open" for _ in range(num_points)]
                    for selpoint in selected_point_numbers:
                        symbols[selpoint] = self.cn.main_plot.instance_symbol
                    self.cn.plotly_pane.object.data[select_trace].marker.symbol = symbols
                elif self.cn.c is not None and configuration_curve_number == select_trace:
                    self.cn.configurations_selected[:] = np.zeros(self.cn.r, dtype=bool)
                    self.cn.configurations_selected[selected_point_numbers] = True
                    self.cn.instance_aggregate_performances[:] = self.cn.performance_matrix.mean(axis=1, where=self.cn.configurations_selected)
                    symbols = [self.cn.main_plot.configuration_symbol + "-open" for _ in range(num_points)]
                    for selpoint in selected_point_numbers:
                        symbols[selpoint] = self.cn.main_plot.configuration_symbol
                    self.cn.plotly_pane.object.data[select_trace].marker.symbol = symbols

                self.uc.update_all()

    def reset_annotations(self, space_name):
        annotations = []
        y_text = self.text_y
        for annotation in self.cn.plotly_pane.object.layout.annotations:
            if annotation.name != space_name:
                annotation_copy = deepcopy(annotation)
                annotation_copy.y = y_text
                annotations.append(annotation_copy)
                y_text = y_text - self.y_space
        return annotations, y_text

    def performance_colour_update(
            self, performance_values, selected: list, curve_number: int,
            annotations: list, attribute_name: str, reset_annotations: bool = True):
        with self.cn.plotly_pane.object.batch_update():
            # TODO: assumes performance is numeric
            if reset_annotations:
                self.cn.plotly_pane.object.layout.annotations = tuple(annotations)
            self.cn.plotly_pane.object.data[curve_number].marker.showscale = True
            self.cn.plotly_pane.object.data[curve_number].marker.color = performance_values
            self.cn.plotly_pane.object.data[curve_number].marker.cmin = min(performance_values[selected])
            self.cn.plotly_pane.object.data[curve_number].marker.cmax = max(performance_values[selected])
            cbdict = self.cn.plotly_pane.object.data[curve_number].marker.colorbar
            if attribute_name is not None:
                self.cn.plotly_pane.object.data[curve_number].marker.colorbar = dict(
                    x=cbdict['x'], y=cbdict['y'], len=cbdict['len'], title=attribute_name, title_side='right')

    def attribute_colour_update(
            self, attribute_values: np.array, curve_number: int, annotations: list, attribute_name: str):
        with self.cn.plotly_pane.object.batch_update():
            # TODO: assumes feature/parameter is numeric
            self.cn.plotly_pane.object.layout.annotations = tuple(annotations)
            self.cn.plotly_pane.object.data[curve_number].marker.showscale = True
            self.cn.plotly_pane.object.data[curve_number].marker.color = attribute_values
            self.cn.plotly_pane.object.data[curve_number].marker.cmin = min(attribute_values)
            self.cn.plotly_pane.object.data[curve_number].marker.cmax = max(attribute_values)
            cbdict = self.cn.plotly_pane.object.data[curve_number].marker.colorbar
            self.cn.plotly_pane.object.data[curve_number].marker.colorbar = dict(
                x=cbdict['x'], y=cbdict['y'], len=cbdict['len'], title=attribute_name, title_side='right')

    def attribute_colour_update_categorical(
            self, attribute_values: np.array, curve_number: int,
            levels: list, level_colours: list, space_name: str,
            annotations: list, y_text: float):
        if len(annotations) > 0:
            y_text = y_text - 0.5 * self.y_space  # TODO: have break_space as parameter?

        with self.cn.plotly_pane.object.batch_update():
            self.cn.plotly_pane.object.data[curve_number].marker.showscale = False
            self.cn.plotly_pane.object.data[curve_number].marker.cmin = None
            self.cn.plotly_pane.object.data[curve_number].marker.cmax = None
            self.cn.plotly_pane.object.data[curve_number].marker.color = attribute_values
            new_annotations = list(
                go.layout.Annotation(
                    xref='paper', x=self.text_x, yref='paper', y=y_text - lev * self.y_space,
                    text='<b>' + levels[lev] + '</b>', showarrow=False, xanchor='left', yanchor='middle',
                    font=dict(color=level_colours[lev]), name=space_name
                ) for lev in range(len(levels))
            )

            self.cn.plotly_pane.object.layout.annotations = tuple(annotations + new_annotations)

    def update_on_selection_change(self):
        annotations = None
        y_text = None
        if self.cn.select_feature.value == 'Aggregate Performance':
            instance_curve_number = [trace.name for trace in self.cn.plotly_pane.object.data].index('Z')
            self.performance_colour_update(
                self.cn.instance_aggregate_performances, self.cn.instances_selected,
                instance_curve_number, annotations, None, reset_annotations=False)
        if self.cn.c is not None and (self.cn.select_parameter.value == 'Aggregate Performance' if self.cn.c is not None else False):
            configuration_curve_number = [trace.name for trace in self.cn.plotly_pane.object.data].index('C')
            self.performance_colour_update(
                self.cn.configuration_aggregate_performances, self.cn.configurations_selected,
                configuration_curve_number, annotations, None, reset_annotations=False)


class CategoricalColours:
    def __init__(self, levels: list, type: str):
        self.levels = levels
        self.type = type

    def get_categorical_colours(self, series: pd.Series):
        if self.type == 'c':
            colour_set = qualitative.Dark24
            level_colours = [colour_set[i] for i in range(len(self.levels))]
        else:
            colour_set = sequential.Plasma
            level_colours = [colour_set[int(round(i * (len(colour_set) - 1) / (len(self.levels) - 1)))] for i in
                             range(len(self.levels))]
        levels_to_colours = dict(zip(self.levels, level_colours))
        colour_vector = series.map(levels_to_colours).values.flatten()
        return self.levels, level_colours, levels_to_colours, colour_vector


class InstanceSpace:
    def __init__(self, com: CombinedSpace, cn: SimpleNamespace):
        self.com = com
        self.cn = cn  # context: metadata, panes, etc.

    # functions for updating
    def update_feature(self, event):
        instance_curve_number = [trace.name for trace in self.cn.plotly_pane.object.data].index('Z')
        annotations, y_text = self.com.reset_annotations("instance")
        if event.new in self.cn.feature_data.columns:
            if self.cn.instance_variable_type[event.new] == 'n':
                self.com.attribute_colour_update(
                    self.cn.feature_data[event.new], instance_curve_number,
                    annotations, event.new
                )
            elif self.cn.instance_variable_type[event.new] == 'c':
                levels = (self.cn.feature_data[event.new].astype('str')).drop_duplicates(keep='first').values.flatten().tolist()
                # FIXME: assumes no ordinal instance features
                levels, level_colours, levels_to_colours, colour_vector = CategoricalColours(levels,
                                                                                             'c').get_categorical_colours(
                    self.cn.feature_data[event.new].astype('str'))
                self.com.attribute_colour_update_categorical(
                    colour_vector, instance_curve_number,
                    levels, level_colours, "instance",
                    annotations, y_text
                )

        elif event.new == "Aggregate Performance":
            self.com.performance_colour_update(
                self.cn.instance_aggregate_performances, self.cn.instances_selected, instance_curve_number,
                annotations, "Aggregate " + self.cn.performance_metric_name, y_text
            )
        elif event.new == "Best Configuration Cluster (Median)":
            self.com.attribute_colour_update_categorical(
                self.cn.isource_colour_vector, instance_curve_number,
                self.cn.isource_levels, self.cn.isource_level_colours, "instance",
                annotations, y_text
            )

    def reset_instances(self, event):
        if event.new is not None:
            trace_names = [trace.name for trace in self.cn.plotly_pane.object.data]
            instance_curve_number = trace_names.index('Z')
            self.cn.instances_selected[:] = np.ones(self.cn.n, dtype=bool)
            self.cn.configuration_aggregate_performances[:] = self.cn.performance_matrix.mean(
                axis=0, where=self.cn.instances_selected.reshape(-1, 1)
            )
            symbols = [self.cn.main_plot.instance_symbol for _ in range(self.cn.n)]
            self.cn.plotly_pane.object.data[instance_curve_number].marker.symbol = symbols
            if self.cn.c is not None:
                configuration_curve_number = trace_names.index('C')
            else:
                configuration_curve_number = None

            self.com.uc.update_all()


class ConfigurationSpace:
    def __init__(self, com: CombinedSpace, cn: SimpleNamespace):
        self.com = com
        self.cn = cn  # context: metadata, panes, etc.

    def update_parameter(self, event):
        configuration_curve_number = [trace.name for trace in self.cn.plotly_pane.object.data].index('C')
        annotations, y_text = self.com.reset_annotations("configuration")
        if event.new in self.cn.parameter_data.columns:
            if self.cn.configuration_variable_type[event.new] == 'n':
                self.com.attribute_colour_update(
                    self.cn.parameter_data[event.new], configuration_curve_number,
                    annotations, event.new
                )
            elif self.cn.configuration_variable_type[event.new] == 'c':
                levels = self.cn.parameter_domain[event.new]
                levels = [str(lev) for lev in levels]
                param_type = self.cn.parameter_type[event.new]
                # assertion disabled
                # assert param_type in ['c', 'o']
                levels, level_colours, levels_to_colours, colour_vector = CategoricalColours(
                    levels, param_type).get_categorical_colours(
                    self.cn.parameter_data[event.new].astype('str')
                )
                self.com.attribute_colour_update_categorical(
                    colour_vector, configuration_curve_number,
                    levels, level_colours, "configuration",
                    annotations, y_text
                )
        elif event.new == "Aggregate Performance":
            self.com.performance_colour_update(
                self.cn.configuration_aggregate_performances, self.cn.configurations_selected,
                configuration_curve_number, annotations, "Aggregate " + self.cn.performance_metric_name, y_text)
        elif event.new == "Configuration Cluster":
            self.com.attribute_colour_update_categorical(
                self.cn.csource_colour_vector, configuration_curve_number,
                self.cn.csource_levels, self.cn.csource_level_colours, "configuration",
                annotations, y_text
            )

    def reset_configurations(self, event):
        if event.new is not None:
            trace_names = [trace.name for trace in self.cn.plotly_pane.object.data]
            instance_curve_number = trace_names.index('Z')
            configuration_curve_number = trace_names.index('C')
            self.cn.configurations_selected[:] = np.ones(self.cn.r, dtype=bool)
            self.cn.instance_aggregate_performances[:] = self.cn.performance_matrix.mean(
                axis=1, where=self.cn.configurations_selected
            )
            symbols = [self.cn.main_plot.configuration_symbol for _ in range(self.cn.r)]
            self.cn.plotly_pane.object.data[configuration_curve_number].marker.symbol = symbols

            self.com.uc.update_all()
