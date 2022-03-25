import re, glob
from typing import List

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_table


app = dash.Dash(__name__)

list_of_files = glob.glob("*.CSV")
match_data = re.compile(r'\-?\d{1,10}\.?d{0,10}')

file_type = re.compile(r'(?P<different>\w+t\dg\w*\.\w+)|(?P<kurs>\w+_\dg\w*\.\w+)|(?P<depth>\w+_[a-zA-Z]+\.\w+)')

annotation_template = "95% от Заданного угла время = {}"
title_template = 'График маневривания по {} KPSI={} KWY={}'
WIDTH = 4

target = {
        'kurs': {
            'tags' : ['dvv','Wy','Pbins','Pzad'],
            'title' : 'курсу',
            'time' : [16.0, 18.0],
            },
        'different': {
            'tags': ['dgp', 'Wz', 'Qbins', 'Qzad'],
            'title': 'дифференту',
            'time' : [28.7, 30.8],
            },
        'depth': {
            'tags': ['dgp', 'Wz', 'Qbins', 'Yizm', 'Yzad'],
            'title': 'глубине',
            }
        }

def calculate_times(df: pd.DataFrame, tags: List, angle:[int, float]) -> tuple:
    maneur_time = df['t'].iloc[next(j for j in range(len(df)) \
                 if df[tags[-1]].iloc[j] == angle)]

    df.loc[(df[tags[-2]] > angle * 0.95) & (df['t'] > maneur_time), 'angle0'] = True

    signal_execution_time = df['t'].iloc[next(j for j in range(len(df)) \
                  if df["angle0"].iloc[j] == True)]
    return (maneur_time, signal_execution_time)

def parse_coeficients(dataset: pd.DataFrame) -> tuple:
    return (dataset.KPSI.iloc[0], dataset.KWY.iloc[0])

def make_figure(dataset: pd.DataFrame, graph_target: tuple) -> go.Figure:
    data = []
    kpsi, kwy = parse_coeficients(dataset)
    target_info, file_name = graph_target

    for tag in target[target_info]['tags']:
        data.append(go.Scatter(
            x = dataset['t'],
            y = dataset[tag],
            name = tag,
            line = dict(width = WIDTH,)
        ))

    figure = go.Figure(data = data)
    figure.update_layout(
            height = 1000,
            title_text = title_template.format(target[target_info]['title'], kpsi, kwy))

    if target_info != 'depth':
        angle = int(match_data.findall(file_name)[0])
        maneur_time, signal_execution_time = calculate_times(dataset, target[target_info]['tags'], angle)
        figure.add_vline(
                x = signal_execution_time,\
                annotation_text = annotation_template.format(signal_execution_time - maneur_time),\
                line_width = 3, \
                annotation=dict(font_size=20, font_family="Times New Roman"), \
                line_dash = "dash", \
                line_color = "red")
    return figure

def super_plotter(list_of_files: List) -> List:
    graph = []
    for file in sorted(list_of_files):
        graph_target = [(k,v) for k,v in file_type.match(file).groupdict().items() if v is not None][0]
        df = pd.read_csv(f'./{file}', sep=r'\s+', skiprows=1)
        try:
            df = df.loc[(df['t'] > target[graph_target[0]]['time'][0]) & \
                (df['t'] < target[graph_target[0]]['time'][1])]
            df.reset_index()
        except KeyError:
            continue
        finally:
            graph.append(dcc.Graph(figure = make_figure(df, graph_target)))
    return graph

app.layout = html.Div(super_plotter(list_of_files))
app.run_server(debug=True, port = 8000)
