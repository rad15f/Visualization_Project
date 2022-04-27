import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


#Return a list containing every occurrence of "ai":

# -- Import and clean data (importing csv into pandas)
pd.options.display.max_columns = 10

# Reading columns
df = pd.read_csv('DataCoSupplyChainDataset.csv', sep=',', encoding='latin-1')

col = df.columns


#Fixing Dates
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)']).dt.date

df = df.sort_values(by = 'order date (DateOrders)')

#Selecting Columns
df_col = df[['Type','Sales per customer', 'Delivery Status','order date (DateOrders)',
       'Late_delivery_risk',  'Category Name', 'Customer City','Customer Country', 'Customer Segment',
       'Customer State',  'Department Name','Market','Order City', 'Order Country','Order Item Discount','Order Item Product Price',
       'Order Item Quantity', 'Sales', 'Order Item Total','Order Profit Per Order', 'Order Region', 'Order State', 'Order Status',
       'Product Name', 'Product Price', 'Shipping Mode']]

df_col = df_col[df_col['Order Status'] != 'SUSPECTED_FRAUD']
df_col = df_col[~df_col['Customer State'].isin(["91732","95758"])]

#Checking for NA
df_col.isnull().sum(axis=0)

#Gettig only Numerical

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical = df_col.select_dtypes(include=numerics)

col_numerical = ['Sales per customer','Order Item Discount',
       'Order Item Product Price', 'Sales','Order Item Total', 'Order Profit Per Order', 'Product Price']


for column in df_col:
       if df_col[column].dtypes=='object':
              print(f"{column},{df_col[column].unique()}")

#Interquantile Range
for x in col_numerical:
       Q1 = np.percentile(df_col[x], 25,
                          interpolation='midpoint')

       Q3 = np.percentile(df_col[x], 75,
                          interpolation='midpoint')
       IQR = Q3 - Q1

       # Above Upper bound
       upper = df_col[x] >= (Q3 + 1.5 * IQR)
       # Below Lower bound
       lower = df_col[x] <= (Q1 - 1.5 * IQR)

       print(x)
       print("Upper bound:")
       print(np.where(upper))
       print("Lower bound:")
       print(np.where(lower))

#Checking for Normality and Outliers
#WE CAN MAKE SUBPLOTS OUT OF EACH ONE OF THIS.
# Boxplot
for x in col_numerical:
       sns.boxplot( x= x, data = df_col)
       plt.title(f'{x} Boxplot')
       plt.show()
# i. Hist-plot # f. Displot
for x in col_numerical:
       sns.displot( x= x, data = df_col, kde= True, bins = 50)
       plt.title(f'{x} Histgoram')
       plt.tight_layout()
       plt.show()
#After further investgiation the column Order Profit Per Order is full of noisy data

sns.regplot( data = numerical, x = 'Order Profit Per Order', y = 'Order Item Discount')
plt.show()

df_col.drop('Order Profit Per Order', inplace = True, axis = 1)
df_col.drop('Late_delivery_risk', inplace = True, axis = 1)

df_col['Profit'] = df_col['Order Item Total'] - df_col['Product Price'] #ORDER ITEM TOTAL INCLUDES DISCOUNT

#SEPARATE DATES

df_col['year'] = pd.DatetimeIndex(df_col['order date (DateOrders)']).year
df_col['month'] = pd.DatetimeIndex(df_col['order date (DateOrders)']).month
df_col['order date (DateOrders)']= pd.to_datetime(df_col['order date (DateOrders)'])
df_col['ym-date'] = df_col['order date (DateOrders)'].dt.strftime('%Y-%m')


#MULTIVARIATE BOXPLOT

sns.boxplot(x="year", y="Sales per customer", data=df_col)
plt.title('Boxplot Sales per customer per Year')
plt.show()

sns.boxplot(x="year", y="Order Item Discount", data=df_col)
plt.show()

sns.boxplot(x="year", y="Order Item Total", data=df_col)
plt.show()

sns.boxplot(x = 'Profit', data= df_col)
plt.show()

sns.histplot(x ='Profit', data =df_col, bins=50)
plt.title('Profit Histogram')
plt.show()
#Digging depper on Order Item Total where it shows a lot of potential outliers


p_outliers = df_col[df_col['Order Item Total'] > 500].copy()
p_outliers.head()
p_outliers['Product Name'].value_counts()


d_outliers = df_col[df_col['Order Item Discount'] > 80].copy()

# PIE CHART & SUBPLOTS LOOKING INTO OUTLIERS

products = df_col[(df_col['Order Item Total'] > 1000) & (df_col['year'] == 2017)].copy()

# ax1.pie(p_outliers['Product Name'].value_counts() ,labels = ['Dell Laptop','Lawn mower','E35 Elliptical','E25 Elliptical','Slope Rangefinder',
# 'Dumbbells'],autopct ='%1.2f%%')
# ax1.axis('square') #make it look prety
# ax1.set_title('Products of Order Total > 500'
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, constrained_layout = True,figsize=(60,10))
ax1.pie(p_outliers['Product Name'].value_counts() ,labels = ['Dell Laptop','Lawn mower','E35 Elliptical','E25 Elliptical','Slope Rangefinder',
'Dumbbells'],autopct ='%1.2f%%')
ax1.axis('square') #make it look prety
ax1.set_title('Products of Order Total > 500')
sns.barplot(x = 'Department Name', y = 'Order Item Total', hue = 'Market',data = p_outliers, ax = ax2)
ax2.set_title('Barplot of Order Item Total vs Department Name by Market')
sns.boxplot(x="Customer Segment", y="Order Item Total",hue = 'year',data=df_col,ax = ax3)
ax3.set_title('Boxplot Order Item Total vs Customer per year')
sns.countplot(x = 'Product Name', data = products, ax = ax4)
ax4.set_title('High Edge Products Name in 2017 ')
plt.show()

#PIE CHART
fig, (ax1) = plt.subplots(1,1, constrained_layout = True,figsize=(16,10))
ax1.pie(df_col['Customer Segment'].value_counts() ,labels = ['Consumer','Corporate','Home Office'],autopct ='%1.2f%%')
ax1.axis('square') #make it look prety
ax1.set_title('Percetange of Sales by Customer Segment ')
plt.show()
# COUNT PLOT

df_col['Type'] = df_col['Type'].replace(['PAYMENT'],'CREDIT')

sns.countplot(x="Type", hue="Customer Country", data= df_col) #change data
plt.title('Countplot by Payment Type per Customer Country')
plt.show()

# CHECKING

orders = df_col[df_col['Order Item Product Price'] != df_col['Sales']]
# HEATMAP
numerical['Profit'] = df_col['Profit']

numerical.drop(['Late_delivery_risk', 'Order Item Quantity'], axis = 1, inplace = True)
plt.figure(figsize=(15,8))
ax = sns.heatmap(numerical.corr(), annot = True)
plt.tight_layout()
plt.show()

#LINEPLOT


plt.figure(figsize=(8,4))
ax = sns.lineplot(data=df_col, x="ym-date", y="Sales", hue = 'Customer Segment')
ax.get_figure().autofmt_xdate()
plt.title('Sales vs Time')
plt.tight_layout()
plt.show()


#BARPLOT GROUP

orders = df_col.groupby('Order Country',as_index = False).sum()
orders = orders.sort_values('Order Item Quantity', ascending = False)

ax = sns.barplot(x="Order Country", y="Order Item Quantity", data=orders.head(10))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title('Order Quantity vs Order Country')
plt.tight_layout()
plt.show()

#BARPLOT STACK

order_segment = df_col.groupby(['Order Country','Customer Segment'],as_index = False).sum()
order_segment = order_segment.sort_values('Order Item Quantity', ascending = False)
order_segment = order_segment[['Order Country','Customer Segment','Order Item Quantity']]

sns.set()
df_pivot = pd.pivot_table(order_segment, index='Order Country', columns='Customer Segment', values='Order Item Quantity', aggfunc='sum')
df_pivot.fillna(0,inplace=True)
df_pivot['Total'] = df_pivot.iloc[:,:].sum(axis = 1)
df_pivot = df_pivot.sort_values('Total', ascending= False)
df_pivot.drop('Total', axis = 1).head(10).plot.bar(stacked=True)
plt.title('Order Quantity vs  Country by Segment')
plt.ylabel('Order Quantity')
plt.tight_layout()
plt.show()

# CATPLOT + VIOLIN
plt.figure(figsize = (15,8))
ax = sns.violinplot(y="Sales", x="Type" , hue = 'Market',data=df_col )
plt.title('Sales vs Payment Typer per Market')
plt.show()

plt.figure(figsize = (15,8))

sns.catplot(y="Sales", x="Type",data=df_col )
plt.title('Sales vs Payment Type Distribution')
plt.tight_layout()
plt.show()


#NORMALITY TEST
from scipy.stats import shapiro
from scipy.stats import normaltest

# normality test
#D’Agostino’s K^2 Test

numerical.drop('Order Profit Per Order',axis = 1, inplace=True)

print('We use the Normality test D’Agostino’s K^2 Test')
for e in numerical:
       stat, p = normaltest(df_col[e])
       print('Statistics=%.3f, p=%.3f' % (stat, p))
       # interpret
       alpha = 0.05
       if p > alpha:
           print(f'{e} looks Gaussian (fail to reject H0)')
       else:
           print(f'{e} Sample does not look Gaussian (reject H0)')


# QQPLOT
from scipy.stats import gamma
from seaborn_qqplot import pplot
pplot(df_col, x="Sales", y=gamma, kind='qq', height=4, aspect=2)
plt.title('QQPLOT of Sales')
plt.tight_layout()
plt.show()


#KERNAL DENSITY ESTIMATE
# sns.kdeplot(data=df_col, x="Order Item Discount",hue = 'Customer Segment')
# plt.show()
#
sns.kdeplot(data=df_col, x="Sales", hue = 'year')
plt.title('Kernel density Sales by Customer Segment')
plt.tight_layout()
plt.show()
#SCATTER PLOT AND REGRESSION LINE

corr ,_ = pearsonr(df_col['Sales per customer'], df_col['Order Item Product Price'])

sns.regplot( data = df_col, y = 'Sales per customer', x = 'Order Item Product Price')
plt.title(f'Sales per customer vs Order Item Product Price, {corr:.2f}')
plt.show()

corr ,_ = pearsonr(df_col['Sales per customer'], df_col['Profit'])
sns.lmplot(data = df_col , y = 'Sales per customer', x = 'Profit')
plt.title(f'Sales per customer vs Profit, {corr:.2f}')
plt.tight_layout()
plt.show()

corr ,_ = pearsonr(df_col['Order Item Discount'], df_col['Sales per customer'])

sns.regplot( data = df_col, x = 'Order Item Discount', y = 'Sales per customer')
plt.title(f'Sales per customer vs Order Item Discount, {corr:.2f}')
plt.show()

corr ,_ = pearsonr(df_col['Product Price'], df_col['Sales per customer'])

sns.regplot( data = df_col, x = 'Product Price', y = 'Sales per customer')
plt.title(f'Sales per customer vs Product Price, {corr:.2f}')
plt.show()

#PCA
pre_pca = numerical.copy()

#STANDARDSCALER
scaler = StandardScaler()
pca = scaler.fit_transform(pre_pca)
pca_t = PCA(n_components=2)
principalcomponents =  pca_t.fit_transform(pca)
x = []
x.append(sum(pca_t.explained_variance_ratio_))
pca_t = PCA(n_components=3)
principalcomponents =  pca_t.fit_transform(pca)
x.append(sum(pca_t.explained_variance_ratio_))
pca_t = PCA(n_components=4)
principalcomponents =  pca_t.fit_transform(pca)
x.append(sum(pca_t.explained_variance_ratio_))
pca_t = PCA(n_components=5)
principalcomponents =  pca_t.fit_transform(pca)
x.append(sum(pca_t.explained_variance_ratio_))

y = [e for e in range(2,6)]

plt.plot(y,x,'go--')
plt.ylabel('Cumulative explained variance')
plt.xlabel('Number of Components')
plt.title('Cumulative explaned variance vs number of components ')
plt.show()

#PCA projection to 2D
pca_t = PCA(n_components=2)
principalcomponents =  pca_t.fit_transform(pca)
df_pca = pd.DataFrame(data = principalcomponents,
                  columns= ['principal component 1', 'principal component 2'])

sns.heatmap(df_pca.corr(),annot = True)
plt.tight_layout()
plt.show()

sum(pca_t.explained_variance_ratio_)
print(df_pca.head())

# PAIRPLOT

sns.pairplot(df_col)
plt.show()
# STATISTICS

df_col.describe()

# a. Line-plot #DONE
# b. Bar-plot : stack, group #DONE
# c. Count-plot #DONE
# d. Cat-plot #DONE
# e. Pie-chart #DONE
# f. Displot #DONE
# g. Pair plot  # DONE
# h. Heatmap #DONE
# i. Hist-plot #DONE
# j. QQ-plot #DONE
# k. Kernal density estimate #DONE
# l. Scatter plot and regression line using sklearn #DONE
# m. Multivariate Box plot #DONE
# n. Area plot ( if applicable)
# o. Violin plot #DONE