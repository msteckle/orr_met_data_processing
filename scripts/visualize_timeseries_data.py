import dash
from dash import dcc, html, callback, no_update
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go

############################################################
# Prep Data
############################################################

# Define the base path and tower names
base = '/home/6ru/Desktop/nsrd_local'
towers = ['TOWA', 'TOWB', 'TOWD', 'TOWF', 'TOWS', 'TOWY']
file = '2017-2022_qa-precheck'

# Read the CSV files into a list of dataframes
tower_data = {}
for tower in towers:
    df = pd.read_csv(f'{base}/{tower}_{file}.csv', index_col=0, parse_dates=True)
    tower_data[tower] = df

############################################################
# Create App
############################################################

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Tower Data Dashboard'),

    # Dropdown for selecting tower and column
    html.Div([
        html.Label('Select Tower:'),
        dcc.Dropdown(
            id='tower-dropdown',
            options=[{'label': tower, 'value': tower} for tower in towers],
            value=towers[0],
            clearable=False
        ),
        html.Label('Select Column:'),
        dcc.Dropdown(
            id='column-dropdown',
            options=[],
            clearable=False
        ),
        html.Button('Add Line', id='add-line-button', n_clicks=0),
    ], style={'margin-bottom': '50px'}),

    # Graph to display selected lines
    dcc.Graph(
        id='line-graph',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
        }
    ),

    # Area to display clicked point datetime
    html.Div([
        html.Label('Clicked Point Datetime:'),
        dcc.Textarea(
            id='click-datetime',
            value='',
            style={'width': '100%', 'height': 50},
            readOnly=True
        ),
    ], style={'margin-top': '20px'}),
])

############################################################
# Define callbacks
############################################################

# Populate columns dropdown when tower is selected
@app.callback(
    Output('column-dropdown', 'options'),
    Output('column-dropdown', 'value'),
    Input('tower-dropdown', 'value')
)
def set_columns_dropdown(selected_tower):
    df = pd.read_csv(f'{base}/{selected_tower}_{file}.csv', nrows=1)  # Load a small sample to get column names
    columns = [{'label': col, 'value': col} for col in df.columns if col != 'datetime']
    return columns, columns[0]['value']

# Update graph with selected lines and handle point click
@app.callback(
    Output('line-graph', 'figure'),
    Output('click-datetime', 'value'),
    Input('add-line-button', 'n_clicks'),
    Input('line-graph', 'clickData'),
    State('tower-dropdown', 'value'),
    State('column-dropdown', 'value'),
    State('line-graph', 'figure')
)
def update_graph_and_display_click_data(n_clicks, clickData, selected_tower, selected_column, existing_figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        return no_update, no_update

    if ctx.triggered[0]['prop_id'] == 'add-line-button.n_clicks':
        if n_clicks == 0:
            return go.Figure(), no_update  # Return an empty figure initially

        df = pd.read_csv(f'{base}/{selected_tower}_{file}.csv', parse_dates=True, index_col=0)

        if selected_column not in df.columns:
            return go.Figure(), no_update  # Return an empty figure if the column is not present

        new_trace = go.Scatter(
            x=df.index,
            y=df[selected_column],
            mode='lines',
            name=f'{selected_tower} - {selected_column}',
            hovertemplate='%{x|%Y-%m-%d %H:%M:%S}<br>%{y}'
        )

        if existing_figure is None or 'data' not in existing_figure:
            fig = go.Figure()
            fig.add_trace(new_trace)
        else:
            fig = go.Figure(existing_figure)
            fig.add_trace(new_trace)

        fig.update_layout(
            title='Tower Data',
            xaxis_title='Datetime',
            yaxis_title='Value',
            clickmode='event+select'
        )

        return fig, no_update

    elif ctx.triggered[0]['prop_id'] == 'line-graph.clickData':
        if clickData is None:
            return no_update, no_update

        point = clickData['points'][0]
        datetime_str = point['x']

        fig = go.Figure(existing_figure)

        # Add a vertical line at the clicked point
        vline = dict(
            type='line',
            x0=datetime_str,
            x1=datetime_str,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='red', width=2, dash='dash')
        )

        # Remove any existing vline annotations to ensure only one vline is present
        fig['layout'].pop('shapes', None)
        fig.add_shape(vline)

        return fig, datetime_str

    return no_update, no_update

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
