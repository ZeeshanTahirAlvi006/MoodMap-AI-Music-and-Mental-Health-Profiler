import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Optional

CHART_STYLE = {
    'figure.facecolor': '#181818',
    'axes.facecolor': '#282828',
    'axes.edgecolor': '#535353',
    'axes.labelcolor': '#FFFFFF',
    'text.color': '#FFFFFF',
    'xtick.color': '#B3B3B3',
    'ytick.color': '#B3B3B3',
    'grid.color': '#535353',
    'grid.alpha': 0.3,
    'font.size': 11,
} # defines the style for the charts

plt.rcParams.update(CHART_STYLE)

METRIC_COLORS = {
    'Anxiety': '#1DB954',
    'Depression': '#1E90FF',
    'Insomnia': '#9B59B6',
    'OCD': '#E74C3C',
} # defines the colors for each metric
METRICS = ['Anxiety','Depression','Insomnia','OCD'] # list of metrics 

def create_genre_bar_chart(df:pd.DataFrame)->Figure: # expected to return Figure
    grouped = df.groupby('genre')[METRICS].mean().reset_index() #mean per genre
    n = len(grouped) # number of genre bars
    x = np.arange(n) # positions of the bars
    width = 0.15 #width of each bar
    fig = Figure(figsize=(12,6),dpi=100)  # built with Figure to allow integration with PyQT5
    ax = fig.add_subplot(111)
    for i,metric in enumerate(METRICS):
        ax.bar(x+(i-1.5)*width,grouped[metric],width,label = metric,color = METRIC_COLORS[metric]) # bar for each metric
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['genre'],rotation = 45,ha='right')
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score by Genre",fontsize = 15,pad=20)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig
    

def create_age_line_chart(df: pd.DataFrame)-> Figure: # returns the figure 
    anxiety_mean = df.groupby('age_group')['Anxiety'].mean().reset_index()
    hours_mean = df.groupby('age_group')['Listening'].mean().reset_index()
    fig= Figure(figsize=(12,6),dpi= 100)
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(anxiety_mean['age_group'],anxiety_mean['Anxiety'],color = METRIC_COLORS['Anxiety'],label = 'Anxiety')
    ax1.set_ylabel('Anxiety Level',color = METRIC_COLORS['Anxiety'])
    ax1.tick_params(axis='y',colors=METRIC_COLORS['Anxiety']) 
    ax1.set_xlabel('Age Group',fontsize=12)
    ax1.set_title("Age vs Anxiety and Listening Time",fontsize = 12,pad=20)

    ax2 = ax1.twinx()
    line2= ax2.plot(hours_mean['age_group'],hours_mean['Listening'],marker="s", color='#1DB954', label="Avg Listening Hours")
    ax2.set_ylabel("Average Listening Hours",color='#1DB954', fontsize=12)
    ax2.tick_params(axis='y',labelcolor='#1DB954')
    lines = line1 + line2
    ax1.legend(lines,[l.get_label() for l in lines],loc='best',fontsize=10)
    fig.tight_layout()
    return fig
    

def create_anxiety_histogram(df:pd.DataFrame,user_score:float)-> Figure:
    fig=Figure(figsize=(12,6),dpi=100)
    ax = fig.add_subplot(111)
    bins = np.arange(0,11)
    if not df.empty and 'Anxiety' in df.columns:
        counts,_,patches = ax.hist(df['Anxiety'],bins = bins,color = METRIC_COLORS['Anxiety'],edgecolor ='white',alpha=0.7)
        idx = max(0, min(int(np.floor(user_score)),len(patches)-1)) #bin index for user score
        if len(patches)> idx >= 0:
            patches[idx].set_facecolor('#E74C3C') #highlight users bin
        percentile = (df['Anxiety']<=user_score).mean()*100
        max_val = max(counts) if len(counts) > 0 else 1
    else:
        percentile =0.0
        max_val =1
    ax.axvline(x = user_score,color='#E74C3C', linestyle='--', linewidth=2.5,
               label=f'Your Score: {user_score:.1f}')
    ax.text(user_score+ 0.15,max_val * 0.08,f"Your Score: {user_score:.1f}\nPercentile: {percentile:.1f}%",
             color='#E74C3C', fontweight='bold',
             bbox=dict(facecolor='#181818', alpha=0.8, edgecolor='#535353'))
    ax.set_xlabel('Anxiety Score')
    ax.set_ylabel("Frequency")
    ax.set_title('Anxiety Score Distribution & Your Percentile')
    ax.set_xlim(-0.5,10.5)
    ax.set_xticks(range(11))
    fig.tight_layout()
    return fig

def create_scatter_plot(df:pd.DataFrame,user_hours:float,user_depression:float)->Figure:
    clean = df[['Listening','Depression']].dropna()
    hours = clean['Listening'].to_numpy()
    depression = clean['Depression'].to_numpy()
    fig = Figure(figsize=(7,5),dpi = 100)
    ax = fig.add_subplot(111)
    ax.scatter(hours,depression,alpha = 0.4,s=20,color = '#1DB954' ,label = 'Survey Respondents')
    if len(hours)>= 2 and np.std(hours)>0:
        m,b = np.polyfit(hours,depression,1)
        x_line = np.linspace(hours.min(),hours.max(),100)
        ax.plot(x_line,m*x_line + b,color='#D9534F', linewidth=2, label=f'Trend (slope={m:.2f})')
    ax.scatter([user_hours],[user_depression],marker='*', s=400,
            color='#F0AD4E', edgecolors='black', linewidths=1.2,
            zorder=5, label='Your data point')  
    ax.set_xlabel('Hours/Day')
    ax.set_ylabel('Depression Score')
    ax.set_title("Listening Hours v/s Depression")
    ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    return fig         

def create_correlation_heatmap(df:pd.DataFrame)->Figure:
    cols = ['Hours per day',"BPM",'Anxiety',"Depression","Insomnia",'OCD']
    corr = df[cols].corr(method ='pearson')
    fig = Figure(figsize=(12,6),dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values,cmap='RdYlGn_r', vmin=-1,vmax=1,aspect ='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols,rotation =45,ha = 'right',fontsize=10)
    ax.set_yticklabels(cols,rotation=45,ha='right', fontsize=10)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i,j]
            text_color = 'white' if abs(val)>0.5 else 'black'
            ax.text(j,i,f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')
    fig.colorbar(im,ax = ax,fraction= 0.046,pad=0.04,label ='Correlation')
    ax.set_title('Correlation Heatmap',fontsize=15,pad=20)
    fig.tight_layout()
    return fig

def create_trend_chart(trend_data:dict,session_count:int)-> Optional[Figure]:
    if session_count < 2:
        return None
    fig = Figure(figsize=(12,6),dpi=100)
    ax = fig.add_subplot(111)
    session = list(range(1,session_count+1))
    for metric in METRICS:
        values = trend_data.get(metric,[])
        if values:
            ax.plot(session[:len(values)],values,marker='o', linewidth=2,
                    color=METRIC_COLORS[metric], label=metric, markersize=6)
    ax.set_xlabel("Session",fontsize = 12)
    ax.set_ylabel("Score",fontsize = 12)
    ax.set_title('Mental Health Trend Across Session',fontsize = 15,pad=20)
    ax.set_ylim(0,10.5)
    ax.legend(fontsize = 10,loc='best')
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    return fig
    
    
    
    

