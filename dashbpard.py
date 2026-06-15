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
    grouped = df.groupby('genre')[METRICS].mean().reset_index() # groups the data by genre , calculates the mean and resets the index
    n = len(grouped) # number of genres bars
    x = np.arange(n) # x positions of the bars
    width = 0.15 # width of the bars
    fig,ax = plt.subplots(figsize=(12,6)) # intializes a new matplot lib plot with figures and axis
    for i, metric in enumerate(METRICS): #loop through each metrics in METRICS
        ax.bar(x+(i-1)*width,grouped[metric],width,label = metric,color = METRIC_COLORS[metric]) # plot the bars for each METRIC
    ax.set_xticks(x) # tells where to put the ticks on X axis
    ax.set_xticklabels(grouped['genre'],rotate= 45,ha = "right") # labels the ticks with genre names
    ax.set_ylabel("Mean Score")  # adds label to Y axis
    ax.set_title("Mean Scores by Genre",fontsize=15,pad = 20) # adds title to the plot
    ax.legend(fontsize = 10) # adds legend to the plot
    plt.tight_layout() # adjust layout to prevent labels from overlapping
    return fig # returns the figure 

def create_age_line_chart(df: pd.DataFrame)-> Figure: # returns the figure 
    anxiety_mean = df.groupby('age_group')['Anxiety'].mean().reset_index() # groups the data by age group , calculates the mean and resets the index
    hours_mean = df.groupby("age_group")["Listening"].mean().reset_index() # groups the data by age group , calculates the mean and resets the index 
    fig, ax1 = plt.subplots(figsize=(12,6)) # creates a new figure and axes 
    line1 = ax1.plot(anxiety_mean["age_group"],anxiety_mean["Anxiety"],marker = "o",color = METRIC_COLORS['Anxiety'],label = "Anxiety Level") # plots the anxiety data 
    ax1.set_ylabel("Anxiety Level",color = METRIC_COLORS['Anxiety'],fontsize = 12) # sets the y axis label 
    ax1.tick_params(axis = "y",labelcolor = METRIC_COLORS['Anxiety']) # sets the y axis tick parameters 
    ax1.set_xlabel("Age Group",fontsize =12) # sets the x axis label 
    ax1.set_title("Age vs Anxiety and Listening Time",fontsize = 15,pad=20) # sets the title 
    ax2 = ax1.twinx() # Create a second y axis that shares the same x axis
    line2 = ax2.plot(hours_mean["age_group"],hours_mean["Listening"],marker = "s",color = '#1DB954',label = "Avg Listening Hours") # plots the listening data
    ax2.set_ylabel("Avg Listening Hours",color = '#1DB954',fontsize = 12) # sets the y axis label 
    ax2.tick_params(axis = "y",labelcolor = '#1DB954') # sets the y axis tick parameters 
    lines = line1 + line2 # combine line and its label
    ax1.legend(lines, [l.get_label() for l in lines],loc = "upper left",fontsize = 10) # adds legend to the plot
    plt.tight_layout() # adjust layout
    return fig # returns the figure 

def create_anxiety_distribution(df:pd.DataFrame,user_score:float)-> Figure:
    fig,ax = plt.subplots(figsize=(12,6)) # creates a new figure and axes 
    bins = np.arange(0,11) # defines the bins for the histogram
    
    if not df.empty and "Anxiety" in df.columns: # checking if the dataframe is empty and has the Anxiety column 
        counts,_,patches = ax.hist(df["Anxiety"],bins = bins,color = METRIC_COLORS['Anxiety'],edgecolor = "white",alpha = 0.7) # creates the histogram 
        idx = max(0, min(int(np.floor(user_score)), len(patches) - 1)) # gets the index of the bin where user score falls 
        if 0 <= idx < len(patches): # checking if the index is within the valid range
            patches[idx].set_facecolor('#E74C3C') # Highlight user score bin in red/orange
        percentile = (df["Anxiety"] <= user_score).mean() * 100 # calculates the percentile 
        max_val = max(counts) if len(counts) > 0 else 1 # finds the maximum value in counts
    else:
        percentile = 0.0 # sets the percentile to 0
        max_val = 1 # sets the maximum value to 1 
        
    # Draw vertical dashed red line at user_score
    ax.axvline(x=user_score, color='#E74C3C', linestyle='--', linewidth=2.5, label=f'Your Score: {user_score:.1f}') # draws a vertical dashed red line at user_score 
    
    # Label the line with the calculated percentile
    ax.text(user_score + 0.15, max_val * 0.8, f"Your Score: {user_score:.1f}\nPercentile: {percentile:.1f}%", # labels the vertical line with the user's score and percentile
            color='#E74C3C', fontweight='bold', bbox=dict(facecolor='#181818', alpha=0.8, edgecolor='#535353')) # adds a textbox to the plot
            
    ax.set_xlabel("Anxiety Score") # sets the x axis label 
    ax.set_ylabel("Frequency") # sets the y axis label 
    ax.set_title("Anxiety Score Distribution & Your Percentile", fontsize=15, pad=20) # sets the title
    ax.set_xlim(-0.5, 10.5) # sets the x axis limits 
    ax.set_xticks(range(11)) # sets the x axis ticks 
    ax.legend(fontsize=10) # adds legend to the plot
    plt.tight_layout() # adjust layout
    return fig # returns the figure
    


