import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from exploratory_data_analysis import df3
sns.set_style('whitegrid')



#Histogram, Probability plot, Box plot

def three_chart_plot(df, feature):

    fig = plt.figure(constrained_layout = True, figsize = (12, 8))
    grid = gridspec.GridSpec(ncols = 3, nrows = 3, figure = fig)

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')

    sns.distplot(df.loc[:, feature], norm_hist = True, ax = ax1)
    plt.axvline(x = df[feature].mean(), c = 'red')
    plt.axvline(x = df[feature].median(), c = 'green')

    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:,feature], plot = ax2)

    ax3 = fig.add_subplot(grid[:, 2])

    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:,feature], ax = ax3 ,orient = 'v')
    
three_chart_plot(df3, 'Yearly Change')
plt.show()




#Correlation Matrix

corrmat = df3.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True)
plt.show()




#Most correlated features-Heat map

corrmat = df3.corr()
top_corr_features = corrmat.index[abs(corrmat["Yearly Change"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df3[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()




#Scatter plot
plt.scatter(y =df3['Yearly Change'],x = df3['Fert. Rate'],c = 'black')
plt.ylabel('Yearly Change')
plt.xlabel('Fert. Rate')
plt.show()






