# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import plotly.graph_objects as go

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df=pd.read_csv(r'C:\Arjun\VIT\Winter Semester 21-22\X - Data Visualization\Project\Placement_Data_Full_Class.csv',encoding='latin-1')
df = df.drop(columns=['sl_no'])
df['salary'].fillna(0, inplace=True)

le = preprocessing.LabelEncoder()
df['degree'] = le.fit_transform(df['degree_t'])

df1 = df
le1 = preprocessing.LabelEncoder()
for col in df:
    if(df[col].dtype=='object'):
        df1[col]=le.fit_transform(df[col])

gapminder=px.data.gapminder()
fig1 = px.scatter(df,x="mba_p",y="etest_p",color="status",facet_col="workex")

fig2 = px.violin(df,y="salary",x="specialisation",color="gender",box=True,points="all")

df_corr = df1.corr() 

fig3 = go.Figure()
fig3.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.index,
        z = np.array(df_corr)
    )
)

df2=pd.read_csv(r'C:\Arjun\VIT\Winter Semester 21-22\X - Data Visualization\Project\Placement_Data_Full_Class.csv',encoding='latin-1')

fig4 = px.bar(df2, x='degree_t', y='status',color="gender", barmode= 'group',pattern_shape="gender", pattern_shape_sequence=[".", "x"])
fig4.update_layout(
    title='Degree Vs Status based on Gedner',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Placed',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1 
)

df1['stat'] = df['status']
fig5 = px.pie(df1, values='workex', names='stat', title='Work Experience Vs Placement')

fig6 = px.scatter(df, x="salary", y="ssc_p", color="gender",size ="etest_p",size_max=20, log_x=True)
fig7 = px.histogram(df, x="salary")
fig8 = px.box(df, y="etest_p")

fig9 = px.bar(df2, x='mba_p', y='status',color = "specialisation")

fig10 = px.treemap(df, path=[px.Constant("status"),'degree_t','specialisation','gender'], values='salary')
fig10.update_traces(root_color="lightgrey")
fig10.update_layout(margin = dict(t=50, l=25, r=25, b=25))

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-1',
        figure=fig1
    ),

    dcc.Graph(
        id='example-graph-2',
        figure=fig2
    ),

    dcc.Graph(
        id='example-graph-3',
        figure=fig3
    ),

    dcc.Graph(
        id='example-graph-4',
        figure=fig4
    ),

    dcc.Graph(
        id='example-graph-5',
        figure=fig5
    ),

    dcc.Graph(
        id='example-graph-6',
        figure=fig6
    ),

    dcc.Graph(
        id='example-graph-7',
        figure=fig7
    ),

    dcc.Graph(
        id='example-graph-8',
        figure=fig8
    ),

    dcc.Graph(
        id='example-graph-9',
        figure=fig9
    ),

    dcc.Graph(
        id='example-graph-10',
        figure=fig10
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
