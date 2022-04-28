# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io

app = Dash(__name__)

df = pd.read_csv(
    r'C:\Arjun\VIT\Winter Semester 21-22\X - Data Visualization\Project\Placement_Data_Full_Class.csv', encoding='latin-1')
df = df.drop(columns=['sl_no'])
df['salary'].fillna(0, inplace=True)

"""
sns.pairplot(df, hue="degree_t")
plt.show()
"""

le = preprocessing.LabelEncoder()
df['degree'] = le.fit_transform(df['degree_t'])
df1 = df
for col in df:
    if(df[col].dtype == 'object'):
        df1[col] = le.fit_transform(df[col])

gapminder = px.data.gapminder()
fig2 = px.scatter(df, x="mba_p", y="etest_p",
                  color="status", facet_col="workex")

fig3 = px.violin(df, y="salary", x="specialisation",
                 color="gender", box=True, points="all")

df_corr = df1.corr()

fig4 = go.Figure()
fig4.add_trace(
    go.Heatmap(
        x=df_corr.columns,
        y=df_corr.index,
        z=np.array(df_corr)
    )
)

df2 = pd.read_csv(
    r'C:\Arjun\VIT\Winter Semester 21-22\X - Data Visualization\Project\Placement_Data_Full_Class.csv', encoding='latin-1')
fig5 = px.scatter(df, x="salary", y="ssc_p", color="gender",
                  size="etest_p", size_max=20, log_x=True)

fig6 = px.histogram(df, x="salary")
fig7 = px.box(df, y="etest_p")

fig8 = px.treemap(df, path=[px.Constant(
    "status"), 'degree_t', 'specialisation', 'gender'], values='salary')
fig8.update_traces(root_color="lightgrey")
fig8.update_layout(margin=dict(t=50, l=25, r=25, b=25))

"""
plt.figure(figsize = (20, 15))
plt.style.use('seaborn-white')

#Specialisation
plt.subplot(234)
ax=sns.countplot(x="specialisation", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("magma", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Work experience
plt.subplot(235)
ax=sns.countplot(x="workex", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("cividis", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Degree type
plt.subplot(233)
ax=sns.countplot(x="degree_t", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("viridis", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12,rotation=20)

#Gender
plt.subplot(231)
ax=sns.countplot(x="gender", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("hot", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Higher secondary specialisation
plt.subplot(232)
ax=sns.countplot(x="hsc_s", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("rocket", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Status of recruitment
plt.subplot(236)
ax=sns.countplot(x="status", data=df, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("copper", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
"""

"""
plt.style.use('seaborn-white')
f,ax=plt.subplots(1,2,figsize=(18,8))
df['workex'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Work experience')
sns.countplot(x = 'workex',hue = "status",data = df)
ax[1].set_title('Influence of experience on placement')
"""

app.layout = html.Div(children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center'
        }
    ),

    html.Div(children=[
        html.H2(
            children='Overview of the Dataset',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-1',
            figure=fig8
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Factors affecting the various attributes',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-2',
            figure=fig4
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Effect of various attributes on salary',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-3',
            figure=fig5
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Analysis of salaries',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-4',
            figure=fig6
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Significance of gender in placements',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-5',
            figure=fig8
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Analyzing salary based on specialization',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-6',
            figure=fig3
        )
    ]),

    html.Div(children=[
        html.H2(
            children='Corelation between Employability Test and MBA',
            style={
                'textAlign': 'center'
            }),
        dcc.Graph(
            id='figure-7',
            figure=fig2
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
