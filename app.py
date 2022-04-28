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
from io import BytesIO
from operator import ge
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import json
import dash.dependencies as dd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import squarify
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud


app = Dash(__name__)

df = pd.read_csv('data.csv', encoding='latin-1')
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

df2 = pd.read_csv('data.csv', encoding='latin-1')
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


plt.style.use('seaborn-white')
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df['workex'].value_counts().plot.pie(
    explode=[0, 0.05], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Work experience')
sns.countplot(x='workex', hue="status", data=df)
ax[1].set_title('Influence of experience on placement')

test_png = 'countplot.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

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
        dcc.Dropdown(
            id="dropdown_1",
            options=["gender", "hsc_s", "degree_t",
                     "specialisation", "workex", "status"],
            clearable=False,
            style={"text-align": "center"}
        ),
        dcc.Graph(
            id='mixedgraph',
            figure={
                'layout': {
                    'font': {
                        'color': 'white'
                    }
                }
            }
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
        dcc.Dropdown(
            id="dropdown",
            options=['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'],
            clearable=False,
            style={"text-align": "center"}
        ),
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
    ]),
    html.Div(children=[
        html.H2(
            children="Influence of experience on placement",
            style={
                'textAlign': 'center'
            }),
        html.Img(src='data:image/png;base64,{}'.format(test_base64),
                     style={'height': '50%', 'width': '50%', "display": "block", "margin-left": "auto", "margin-right": "auto",
                            "width": "50%"}),
    ])
])


@app.callback(
    Output("figure-3", "figure"),
    Input("dropdown", "value"))
def update_bar_chart(year):
    fig5 = px.scatter(df, x="salary", y=year, color="gender",
                      size="etest_p", size_max=20, log_x=True,)
    return fig5


@app.callback(
    Output("mixedgraph","figure"),
    Input("dropdown_1","value"))
def update_mixedgraph(value):
    if(value=="gender"):
        l=list(df['gender'].value_counts())
        dg=pd.DataFrame({"Gender":["Male","Female"],"Count":l})
        fig=px.bar(dg,x="Gender",y="Count",color_discrete_sequence=px.colors.sequential.RdBu,)
        return fig
    elif(value=="degree_t"):
        l=list(df['degree_t'].value_counts())
        dg=pd.DataFrame({"degree_t":["Comm & Mgmt","Sci & Tech","Others"],"Count":l})
        fig=px.bar(dg,x="degree_t",y="Count",color_discrete_sequence=px.colors.sequential.RdBu, )
        return fig
    elif(value=="hsc_c"):
        l=list(df['hsc_s'].value_counts())
        dg=pd.DataFrame({"hsc_c":["Commerce","Science","Arts"],"Count":l})
        fig=px.bar(dg,x="hsc_c",y="Count",color_discrete_sequence=px.colors.sequential.RdBu, )
        return fig    
    elif(value=="specialisation"):
        l=list(df['specialisation'].value_counts())
        dg=pd.DataFrame({"specialisation":["Mkt&Fin","Mkt&HR"],"Count":l})
        fig=px.bar(dg,x="specialisation",y="Count",color_discrete_sequence=px.colors.sequential.RdBu, )
        return fig
    elif(value=="workex"):
        l=list(df['workex'].value_counts())
        dg=pd.DataFrame({"workex":["No","Yes"],"Count":l})
        fig=px.bar(dg,x="workex",y="Count",color_discrete_sequence=px.colors.sequential.RdBu, )
        return fig
    else:
        l=list(df['status'].value_counts())
        dg=pd.DataFrame({"status":["Placed","Not Placed"],"Count":l})
        fig=px.bar(dg,x="status",y="Count",color_discrete_sequence=px.colors.sequential.RdBu, )   
        return fig 
        

if __name__ == '__main__':
    app.run_server(debug=True)
