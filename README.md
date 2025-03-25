import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings #ignore unwanted warnings

warnings.simplefilter('ignore')

client_data = pd.read_csv('client_data.csv')
price_data = pd.read_csv('price_data.csv')

client_data.head(3)

price_data.head(2)

client_data.info()

price_data.info()

#checking duplicates
print(client_data[client_data.duplicated()])
print(price_data[price_data.duplicated()])

client_data.describe()

client_data.shape

print('Number of Unique Clinets: ',price_data.id.nunique())

price_data.describe()

price_data.shape

#Merging Client & Price dataset using Customer  ID
client_churn_info = client_data[['id','churn']]
price_df = client_churn_info.merge(price_data,on='id')
price_df.head()

#Plotting histogram to see the distribution of the data using the mean values of both clinet and price datasets
price_df.groupby(['id','price_date']).mean().hist(figsize=(20,10))
plt.show()

#Since the price date is in object state, convert it into date format
#Changing datatype : price date => object -> datetime64
price_data = price_data.astype({'price_date' : 'datetime64[ns]'})

#Plotting both Energy and Power prices as per the CHURN category by price dated months
churn_grp_price = price_df[price_df['churn']==1].groupby(['price_date'])[['price_off_peak_var','price_peak_var','price_mid_peak_var','price_off_peak_fix','price_peak_fix','price_mid_peak_fix']].mean()
non_churn_grp_price = price_df[price_df['churn']==0].groupby('price_date')[['price_off_peak_var','price_peak_var','price_mid_peak_var','price_off_peak_fix','price_peak_fix','price_mid_peak_fix']].mean()

#Plotting average price of energy by month
plt.figure(figsize=(15,3))
plt.xticks(rotation=45)
plt.subplot(131)
non_churn_grp_price.price_off_peak_var.plot()
churn_grp_price.price_off_peak_var.plot()
plt.xticks(rotation=45)
plt.legend(['Not Churn','Churn'])
plt.title('Power price at off peak')
plt.subplot(132)
non_churn_grp_price.price_peak_var.plot()
churn_grp_price.price_peak_var.plot()
plt.legend(['Not Churn','Churn'])
plt.title('Power price at peak')
plt.xticks(rotation=45)
plt.subplot(133)
non_churn_grp_price.price_mid_peak_var.plot()
churn_grp_price.price_mid_peak_var.plot()
plt.legend(['Not Churn','Churn'])
plt.title('Power price at mid peak')
plt.xticks(rotation=45)
plt.suptitle('Power price of Non-Churn vs Churn Customers')
plt.subplots_adjust(top=0.8)
plt.show()


#Replotting by the average price of power through month
plt.figure(figsize=(15,3))
plt.xticks(rotation=45)
plt.subplot(131)
non_churn_grp_price.price_off_peak_fix.plot()
churn_grp_price.price_off_peak_fix.plot()
plt.xticks(rotation=45)
plt.legend(['Not Churn','Churn'])
plt.title('Power price at off peak')
plt.subplot(132)
non_churn_grp_price.price_peak_fix.plot()
churn_grp_price.price_peak_fix.plot()
plt.legend(['Not Churn','Churn'])
plt.title('Power price at peak')
plt.xticks(rotation=45)
plt.subplot(133)
non_churn_grp_price.price_mid_peak_fix.plot()
churn_grp_price.price_mid_peak_fix.plot()
plt.legend(['Not Churn','Churn'])
plt.title('Power price at mid peak')
plt.xticks(rotation=45)
plt.suptitle('Power price of Non-Churn vs Churn Customers')
plt.subplots_adjust(top=0.8)
plt.show()

pd_corr = price_data.corr(numeric_only=True)
mask = np.triu(np.ones_like(pd_corr))
sns.heatmap(pd_corr,annot=True,cmap="crest",linewidth=.5,mask=mask)
plt.title('Correlation Plot')
plt.show()

#Since there are multiple non-necessary price values, let's sort them using the above data of Heatmap
price_data.drop(['price_peak_var','price_peak_fix','price_mid_peak_var'],axis=1,inplace=True)

#Filtering out the January and December energy off peak price
price_off_peak_energy = price_data[['id','price_off_peak_var']]
jan_prices = price_off_peak_energy.groupby('id').price_off_peak_var.first().reset_index().rename(columns={'price_off_peak_var':'price_off_peak_var_jan'})
dec_prices = price_off_peak_energy.groupby('id').last().price_off_peak_var.reset_index().rename(columns={'price_off_peak_var':'price_off_peak_var_dec'})

price_data.drop('price_off_peak_var',axis=1,inplace=True)
#Taking average of Power off-peak and mid-peak
price_data = price_data.groupby('id').mean().reset_index()

price_data = price_data.merge(jan_prices,on='id').merge(dec_prices,on='id')
price_data['energy_off_peak_variation'] = price_data.price_off_peak_var_jan - price_data.price_off_peak_var_dec
price_data.drop(['price_off_peak_var_jan','price_off_peak_var_dec'],axis=1,inplace=True)

#Final price dataset
price_data.head()

#Changing hashed values to meaningful labels for easy understanding in both sales channels and in origin campaign
print('Unique Sales Channels : \n',client_data.channel_sales.unique())
print('\nUnique Origin Campaign : \n',client_data.origin_up.unique())


channel_mask = {
    'MISSING':'missing_data',
    'foosdfpfkusacimwkcsosbicdxkicaua':'channel_1',
    'lmkebamcaaclubfxadlmueccxoimlema':'channel_2',
    'usilxuppasemubllopkaafesmlibmsdf':'channel_3',
    'ewpakwlliwisiwduibdlfmalxowmwpci':'channel_4',
    'epumfxlbckeskwekxbiuasklxalciiuu':'channel_5',
    'sddiedcslfslkckwlfkdpoeeailfpeds':'channel_6',
    'fixdbufsefwooaasfcxdxadsiekoceaa':'channel_7',
}
origin_mask = {
    'lxidpiddsbxsbosboudacockeimpuepw' : 'origin_1',
    'kamkkxfxxuwbdslkwifmmcsiusiuosws' : 'origin_2',
    'ldkssxwpmemidmecebumciepifcamkci' : 'origin_3',
    'usapbepcfoloekilkwsdiboslwaxobdp' : 'origin_4',
    'ewxeelcelemmiwuafmddpobolfuxioce' : 'origin_5',
    'MISSING' : 'origin_missing'
}
client_data.replace({
    'has_gas' : {
        't':1,'f':0
    },
    'channel_sales':channel_mask,
    'origin_up':origin_mask,
},inplace=True)
#Final client dataset
client_data.head(3)


df = client_data.merge(price_data,on='id')
print('Total No of Clients in Price Dataset : ',price_data.id.nunique(),'\nTotal No of Clients in Client Dataset : ',client_data.id.nunique(),'\nTotal No of Clients after merging : ',df.id.nunique())


fig, ax = plt.subplots()
bottom = np.zeros(1)

values = {
    "not churned":np.array([client_data.churn.value_counts()[0] /(client_data.churn.value_counts()[0]+client_data.churn.value_counts()[1])]),
    "churned":np.array([client_data.churn.value_counts()[1] /(client_data.churn.value_counts()[0]+client_data.churn.value_counts()[1])])
}
bottom=[0]

for boolean, weight_count in values.items():
    print(boolean, weight_count)
    p = ax.bar(["x-axis"],weight_count, 0.5, label=boolean, bottom=bottom, align='center')
    ax.annotate(round(weight_count[0],3), ('x-axis',1-weight_count[0]+0.03), color='white')
    bottom+=weight_count
ax.set_title("Checking the balance of the dataset")
ax.legend(loc='upper right')


channel = client_data[['churn','id','channel_sales']].groupby(['channel_sales','churn']).count().unstack().fillna(0)
channel = channel.div(channel.sum(axis=1), axis=0).mul(100)
channel.plot(kind='bar', stacked=True, title='Channel wise churn')


df.hist(figsize=(15,10))
plt.show()


ax = df.hist(figsize=(15,10),column=['cons_12m', 'cons_gas_12m', 'imp_cons', 'margin_net_pow_ele', 'net_margin',
       'pow_max',  'price_mid_peak_fix','energy_off_peak_variation','price_off_peak_fix'],bins=10)
plt.show()


def annotation_labeling(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)

plt.pie(df.churn.value_counts(),
        labels=['Non Churn','Churn'],
        autopct=lambda x : annotation_labeling(x,df.churn.value_counts().values))
plt.title('Churn VS Non-Churn Clients')
plt.show()


def annotation_labeling(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# Assuming 'df' is your DataFrame containing all client data
# and 'churn' is a column indicating churn status (1 for churn, 0 for non-churn)

# Create DataFrames for churned and non-churned clients
churn_data = df[df['churn'] == 1]
non_churn_data = df[df['churn'] == 0]

plt.figure(figsize=(10,5))

plt.subplot(122)
plt.title('Churn Client')
_,_,lbl_text = plt.pie(non_churn_data.origin_up.value_counts(),
        labels=non_churn_data.origin_up.value_counts().index,
        autopct=lambda x : annotation_labeling(x,non_churn_data.origin_up.value_counts())) #,colors=colors) remove colors

plt.subplot(121)
plt.title('Non Churn Client')
_,_,lbl_text = plt.pie(churn_data.origin_up.value_counts(),
        labels=churn_data.origin_up.value_counts().index,
        autopct=lambda x : annotation_labeling(x,churn_data.origin_up.value_counts())) #,colors=colors) remove colors

plt.suptitle("Origin Campaigns")
plt.show()



clients_channel_count = df['channel_sales'].value_counts()
data = {
    'Channel Name' : clients_channel_count.index.values,
    'Channel Count' : clients_channel_count.values,
    'Percentage' : (clients_channel_count.values/client_data.shape[0])*100
}
channel_sales_data = pd.DataFrame(data)


def annotation_labeling(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


clients_channel_count = df['channel_sales'].value_counts()
data = {
    'Channel Name' : clients_channel_count.index.values,
    'Channel Count' : clients_channel_count.values,
    'Percentage' : (clients_channel_count.values/client_data.shape[0])*100
}
channel_sales_data = pd.DataFrame(data)


churn_channel_sales =churn_data['channel_sales'].value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))
plt.figure(figsize=(10,5))


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0','#ffb3e6']

_,_,ax1_text = ax1.pie(x=channel_sales_data['Channel Count'][:-3],
                       labels=channel_sales_data['Channel Name'][:-3],
                       autopct = lambda x : annotation_labeling(x,channel_sales_data['Channel Count'].values),
                       colors=colors) # Use the defined 'colors' variable
ax1.set_title('Total Channel Sales Count')
_,_,ax2_text=ax2.pie(x=churn_channel_sales.values,labels=churn_channel_sales.index,autopct = lambda x : annotation_labeling(x,churn_channel_sales.values),colors=colors)
ax2.set_title('Churn Channel Sales Count')

plt.show()




Principal Component Analysis

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


for col in df.select_dtypes(include=['datetime64']).columns:
    df[col] = df[col].astype(np.int64) // 10**9

df = pd.get_dummies(df, drop_first=True)
x, y = df.drop('churn', axis=1), df.churn
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(StandardScaler().fit_transform(x)), columns=['PCA1', 'PCA2'])
pca_df['churn'] = df['churn']
pca_df.head()


plt.figure(figsize=(15,5))
plt.subplot(131)
sns.scatterplot(data=pca_df[pca_df['churn']==1],x='PCA1',y='PCA2')
plt.title('Chrun Plot')
plt.subplot(132)
sns.scatterplot(data=pca_df[pca_df['churn']==0],x='PCA1',y='PCA2')
plt.title('Non Chrun Plot')
plt.subplot(133)
sns.scatterplot(data=pca_df,x='PCA1',y='PCA2',hue='churn')
plt.title('PCA plot')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.27,stratify=y,random_state=42)
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_train_pred = dt_model.predict(x_train)
y_pred = dt_model.predict(x_test)

cm_pred2 = confusion_matrix(y_test,y_pred,labels = dt_model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm_pred2,display_labels=dt_model.classes_).plot()
plt.show()

print("Model's f1 score for training dataset :",f1_score(y_train,y_train_pred),
      "\nModel's f1 score for test dataset :",f1_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_train_pred = dt_model.predict(x_train)
y_pred = dt_model.predict(x_test)

cm_pred2 = confusion_matrix(y_test,y_pred,labels = dt_model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm_pred2,display_labels=dt_model.classes_).plot()
plt.show()

print("Model's f1 score for training dataset :",f1_score(y_train,y_train_pred),
      "\nModel's f1 score for test dataset :",f1_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


#Feature distribution
x.hist(figsize=(15,15),bins=10)
plt.show()


from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
merged_data = pd.merge(client_data, price_data, on="id", how="inner")
label_encoder = LabelEncoder()
for column in ['channel_sales', 'has_gas', 'origin_up']:
    merged_data[column] = label_encoder.fit_transform(merged_data[column])
features = merged_data.drop(columns=['id', 'price_date', 'churn', 'date_activ', 'date_end',
                                      'date_modif_prod', 'date_renewal'])
target = merged_data['churn']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='viridis')
plt.title("Logistic Regression")
plt.show()


merged_data = pd.merge(client_data, price_data, on="id", how="inner")
x_train, x_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)

print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)


from sklearn.svm import SVC #Import SVC from the correct module
svc_model = SVC(random_state=0).fit(x_train, y_train)
yt_pred, y_pred = svc_model.predict(x_train), svc_model.predict(x_test)
cm_pred2 = confusion_matrix(y_test, y_pred, labels=svc_model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm_pred2, display_labels=svc_model.classes_).plot()
plt.title("Support Vector Machine")
plt.show()

print("Model's f1 score for training dataset:", f1_score(y_train, yt_pred))
print("Model's f1 score for test dataset:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X_train, y_train)

predicted_y_knn = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test, y_test)

conf_matrix_knn = confusion_matrix(y_test, predicted_y_knn)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix_knn, display_labels=knn_model.classes_).plot(cmap=plt.cm.Blues)

print("KNN Accuracy:", accuracy_knn)
print("KNN Classification Report:")
print(classification_report(y_test, predicted_y_knn))

plt.title("KNN Confusion Matrix")
plt.show()


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=0).fit(x_train, y_train)
yt_pred, y_pred = rf_model.predict(x_train), rf_model.predict(x_test)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                       display_labels=rf_model.classes_).plot()
plt.title("Random Forest")
plt.show()

print("Model's f1 score for training dataset:", f1_score(y_train, yt_pred))
print("Model's f1 score for test dataset:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_curve, roc_auc_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_probs = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Random Forest ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


from sklearn import tree
dt_model = tree.DecisionTreeClassifier().fit(x_train, y_train)
y_train_pred, y_pred = dt_model.predict(x_train), dt_model.predict(x_test)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                       display_labels=dt_model.classes_).plot()
plt.title("Decision tree")
plt.show()

print(f"Model's f1 score for training dataset: {f1_score(y_train, y_train_pred)}")
print(f"Model's f1 score for test dataset: {f1_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

y_pred_ann = (model.predict(X_test) > 0.5).astype(int)
accuracy_ann = model.evaluate(X_test, y_test, verbose=0)[1]

print("ANN Accuracy:", accuracy_ann)
print("ANN Classification Report:")
print(classification_report(y_test, y_pred_ann))

conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix_ann, display_labels=[0, 1]).plot(cmap=plt.cm.Blues)
plt.title("ANN Confusion Matrix")
plt.show()


import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = xgb_model.score(X_test, y_test)
print("XGBoost Accuracy:", accuracy_xgb)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb, display_labels=xgb_model.classes_).plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Now XGBClassifier is defined
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)
xgb_model.fit(X_train, y_train)

y_probs = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()



import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=rf_model.classes_).plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - start_time

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = xgb_model.score(X_test, y_test)
print("XGBoost Accuracy:", accuracy_xgb)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb, display_labels=xgb_model.classes_).plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()

print(f"Random Forest Training Time: {rf_train_time:.4f} seconds")
print(f"XGBoost Training Time: {xgb_train_time:.4f} seconds")

print("\nModel Comparison Summary:")
if accuracy_rf > accuracy_xgb:
    print("Random Forest performs better in terms of accuracy.")
elif accuracy_rf < accuracy_xgb:
    print("XGBoost performs better in terms of accuracy.")
else:
    print("Both models have the same accuracy.")

if rf_train_time < xgb_train_time:
    print("Random Forest is faster in training.")
else:
    print("XGBoost is faster in training.")



Random Forest Training Time: 4.5922 seconds
XGBoost Training Time: 0.4476 seconds

Model Comparison Summary:
Random Forest performs better in terms of accuracy.
XGBoost is faster in training.
