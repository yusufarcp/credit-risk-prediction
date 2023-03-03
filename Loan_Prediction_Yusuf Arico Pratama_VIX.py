#!/usr/bin/env python
# coding: utf-8

# Nama: Yusuf Arico Pratama<br>
# VIX: Data Science

# # Latar Belakang Tugas

# Sebagai tugas akhir dari masa kontrakmu sebagai intern Data Scientist di ID/X Partners, kali ini kamu akan dilibatkan dalam projek dari sebuah lending company. Kamu akan berkolaborasi dengan berbagai departemen lain dalam projek ini untuk menyediakan solusi teknologi bagi company tersebut. ***Kamu diminta untuk membangun model yang dapat memprediksi credit risk menggunakan dataset yang disediakan oleh company yang terdiri dari data pinjaman yang diterima dan yang ditolak.*** Selain itu kamu juga perlu mempersiapkan media visual untuk mempresentasikan solusi ke klien. Pastikan media visual yang kamu buat jelas, mudah dibaca, dan komunikatif. Pengerjaan end-to-end solution ini dapat dilakukan di Programming Language pilihanmu dengan tetap mengacu kepada framework/methodology Data Science.

# # Tujuan

# ***Credit risk prediction*** adalah cara yang efektif untuk mengevaluasi apakah calon peminjam akan melunasi pinjaman, khususnya dalam pinjaman peer-to-peer di mana masalah ketidakseimbangan kelas yang lazim terjadi.

# # Import library

# In[1]:


# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Preparation

# In[2]:


# Read and load data into dataframe
loan_data = pd.read_csv('loan_data_2007_2014.csv')
loan_data.head()


# In[3]:


# check the data size
loan_data.shape


# ## Define Target Variable / Labeling

# In[4]:


# Cek kolom 'loan_status'. loan_status akan menjadi data untuk target prediksi
loan_data.loan_status.value_counts()


# In[5]:


# Mendapatkan proporsi dari nilai observasi dari nilai unik setiap variabel
loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()


# Karena kita ingin memprediksi apakah suatu pinjaman bersifat beresiko atau tidak, maka kita perlu mengetahui histori akhir dari setiap jenis pinjaman apakah pinjaman tersebut gagal bayar / ditagih, atau lunas. Selanjutnya, kita akan mengklasifikasikan pinjaman tersebut sebagai ***good loans*** (tidak berisiko) dan ***bad loans*** (berisiko).
# 
# - good loans = **['Current', 'Fully Paid', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid']**
# - bad loans = **['Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']**

# In[6]:


# definikan nilai
good_loans = ['Current', 'Fully Paid', 'In Grace Period', 
              'Does not meet the credit policy. Status:Fully Paid']

# membuat kolom baru untuk proses klasifikasi
loan_data['good_bad_loan'] = np.where(loan_data['loan_status'].isin(good_loans), 1, 0)


# In[7]:


# check balance
plt.title('Good (1) vs Bad (0) Loans Balance')
sns.barplot(x=loan_data.good_bad_loan.value_counts().index,y=loan_data.good_bad_loan.value_counts().values)


# ## Data Pre-processing, Cleaning, and Feature Engineering
# 
# **Drop columns:**
# - Drop kolom 'Unnamed: 0' yang merupakan salinan dari nilai index.
# - Drop the columns having > 50% missing values. (columns with 0 unique value are also columns that have 100% missing value)
# - Drop kolom yang mempunyai > 50% *missing values*. (kolom bernilai 0 juga temasuk kolom yang mempunyai *missing value*)
# - Drop kolom 'application_type' dan 'policy_code' (karena hanya mempunyai 1 *unique value*).
# - Drop *identifier columns:* id, member_id, title, emp_title, url, zip_code, desc, policy_code (Tidak bisa digunakan untuk membuat model).
# - Drop sub_grade, karena memiliki informasi yang sama dengan kolom grade columns.

# In[8]:


# Displays column names, complete (non-missing) cases per column, and datatype per column.
loan_data.info()


# In[9]:


# get a list of columns that have more than 50% null values
na_values = loan_data.isnull().mean()
na_values[na_values>0.5]


# In[10]:


# Filtering data with less than 2 unique values
loan_data.nunique()[loan_data.nunique() < 2].sort_values()


# In[11]:


# Drop the irrelevant columns
loan_data.drop(['Unnamed: 0', 'desc', 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 
                'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m', 'open_il_6m', 
                'open_il_12m', 'open_il_24m','mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
                'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'policy_code', 
                'application_type','id', 'member_id', 'sub_grade', 'emp_title', 'url', 'title', 
                'zip_code'], axis=1, inplace = True)


# ### Data Understanding and Data Leakage

# Pemahaman data/kolom itu penting. Saya ingin memprediksi apakah suatu pinjaman berisiko atau tidak, sebelum saya berinvestasi dalam pinjaman tersebut, bukan setelahnya. Masalah dengan data yang ada di kolom yang terkait dengan status pinjaman saat ini. Saya hanya bisa mendapatkan data dari kolom tersebut setelah pinjaman dikeluarkan, dengan kata lain, setelah saya berinvestasi dalam pinjaman.
# 
# - Kolom terkait status pinjaman saat ini (setelah diterbitkan): **[['issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'pemulihan', 'collection_recovery_fee', 'last_pymnt_d ', 'last_pymnt_amnt', 'next_pymnt_d']]**
# 
# Misal, **'out_prncp'** (outstanding principal (Sisa sisa pokok pinjaman untuk total dana yang didanai)), bila out_prncp adalah 0, maka berarti pinjaman sudah lunas, mudah diprediksi berdasarkan variabel yang satu ini saja, dan akan menjadi sangat akurat. <br>
# Contoh lain adalah dengan 'pemulihan', pemulihan hanya terjadi setelah peminjam tidak mampu membayar pinjaman dan lembaga pemberi pinjaman memulai proses pemulihan pinjaman. Tentu kita tahu bahwa pinjaman itu buruk dan berisiko, hanya dari info ini saja. Variabel-variabel tersebut dapat diprediksi dengan sangat akurat karena sudah terjadi.
# 
# Dalam ilmu data, variabel semacam ini disebut Kebocoran Data. Kebocoran Data ***(Data Leakage)*** adalah pembuatan informasi tambahan yang tidak terduga dalam data pelatihan, yang memungkinkan model atau algoritme pembelajaran mesin membuat prediksi yang tidak realistis. Ini adalah data yang tidak akan kami dapatkan saat kami menggunakan model dalam penerapan. Kami tidak akan tahu apakah akan ada biaya pemulihan, atau apakah pokok pinjaman akan menjadi 0 atau tidak sebelum pinjaman selesai. Kami tidak akan mendapatkan data tersebut sebelum kami berinvestasi dalam pinjaman.
# 
# Jadi, kolom yang mengandung Kebocoran Data akan **dihapus** dan hanya menyimpan kolom dengan data yang bisa diperoleh sebelum pinjaman diinvestasikan.

# In[12]:


leakage_col = ['issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
                'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d']

loan_data.drop(columns=leakage_col, axis=1, inplace=True)


# ### Korelasi antar fitur variabel

# In[13]:


#Check correlation
plt.figure(figsize=(24,24))
sns.heatmap(loan_data.corr(), annot=True, annot_kws={'size':14})


# **Catatan** : loan_amnt, funded_amnt, funded_amnt_inv mempunyai nilai korelasi yang hampir sama dengan kolom lain. Jadi mungkin memiliki data yang bersifat duplikat.

# In[14]:


# Check the suspect similar columns
loan_data[['loan_amnt','funded_amnt','funded_amnt_inv']].describe()


# Berdasarkan data diatas, dapat disimpulkan bahwa ketiga fitur tersebut memiliki data yang hampir sama, jadi kita bisa menghapus dua fiturnya.

# In[15]:


loan_data.drop(columns = ['funded_amnt', 'funded_amnt_inv'], inplace = True)


# ### Checking missing value

# In[16]:


loan_data.isnull().sum()


# In[17]:


# retriving the columns which has any null values
loan_data_columns=loan_data.columns[loan_data.isnull().any()].tolist()
loan_data[loan_data_columns].isnull().sum()*100/len(loan_data)


# **Notes: [tot_coll_amt, tot_cur_bal, total_rev_hi_lim]** mempunya nilai total missing sebesar 15%. Jadi ketiga kolom tersebut perlu kita cek terlebih dahulu.
# - tot_coll_amt: Jumlah total tagihan yang pernah terhutang
# - tot_cur_bal: Total saldo saat ini dari semua akun
# - total_rev_hi_lim: Total kredit/batas kredit tinggi bergulir

# In[18]:


# Check tot_coll_amt, tot_cur_bal, total_rev_hi_lim
total_cols = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

loan_data[total_cols].head(10)


# In[19]:


loan_data[total_cols].sample(10)


# In[20]:


loan_data[total_cols].describe()


# In[21]:


loan_data.boxplot(column=['tot_coll_amt'])
plt.show()


# In[22]:


loan_data.boxplot(column=['tot_cur_bal'])
plt.show()


# In[23]:


loan_data.boxplot(column=['total_rev_hi_lim'])
plt.show()


# Kesimpulan:
# - 75% data dari kolom tot_coll_amt adalah 0.
# - Data untuk setiap baris sangat berbeda sehingga tidak mungkin mengisi nilai yang hilang dengan nilai rata-rata atau nilai lainnya..
# - total missing value 70276 = 15.07% dari seluruh data.
# - Baris dari nilai yang hilang di kolom tersebut akan dihapus.

# In[24]:


# Menghapus nilai missing value
loan_data.dropna(subset = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'], inplace = True)
loan_data.reset_index(drop= True, inplace = True)


# ### Pre-processing Beberapa variabel kontinyu (Data Type Transformation)
# Variabel berikut tidak memiliki tipe data yang sesuai dan harus dimodifikasi.

# In[25]:


continuous_cols = ['term', 'emp_length', 'earliest_cr_line', 'last_credit_pull_d']
loan_data[continuous_cols]


# #### 1. term
# term: Jumlah pembayaran pinjaman. Nilai dalam bulan dan dapat berupa 36 atau 60.

# In[26]:


# Check the data
loan_data['term']


# In[27]:


# Convert to numerical datatype and replace months with empty strng
loan_data['term'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))
loan_data['term']


# #### 2. emp_length
# Lama waktu bekerja dalam tahun. Nilai yang mungkin adalah antara 0 dan 10 di mana 0 berarti kurang dari satu tahun dan 10 berarti sepuluh tahun atau lebih.

# In[28]:


# Displays unique values of emp_length
loan_data['emp_length'].unique()


# In[29]:


emp_map = {
    '< 1 year' : '0',
    '1 year' : '1',
    '2 years' : '2',
    '3 years' : '3',
    '4 years' : '4',
    '5 years' : '5',
    '6 years' : '6',
    '7 years' : '7',
    '8 years' : '8',
    '9 years' : '9',
    '10+ years' : '10'
}

loan_data['emp_length'] = loan_data['emp_length'].map(emp_map).fillna('0').astype(int)
loan_data['emp_length'].unique()


# ####  3. earliest_cr_line
#  earliest_cr_line: Bulan batas kredit peminjam yang paling awal dilaporkan dibuka

# In[30]:


# # Displays a column
loan_data['earliest_cr_line']


# In[31]:


# Extracts the date and the time from a string variable that is in a given format
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')


# In[32]:


# Asusmsi kan kita berada di bulan Desember 2017
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
# hitung selisih antara dua tanggal dalam bulan, ubah menjadi tipe data numerik dan bulatkan, setelah itu simpan dalam variabel baru.


# In[33]:


# Tunjukan sedikit statistika deskriptif dari kolom ini
loan_data['mths_since_earliest_cr_line'].describe()


# In[34]:


# Menampilkan baris yang memiliki nilai negatif
loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]


# **Catatan:** Tanggal dari tahun 1969 dan sebelumnya tidak dapat dikonversi dengan benar dan mempunyai nilai negatif yang berbeda.

# In[35]:


# Mengubah dtype menjadi string dan mengubah tahun 2069 menjadi 1969 dan seterusnya
loan_data['earliest_cr_line_date'] = loan_data['earliest_cr_line_date'].astype(str)
loan_data['earliest_cr_line_date'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['earliest_cr_line_date'][loan_data['mths_since_earliest_cr_line'] < 0].str.replace('20','19')


# In[36]:


# Pengecekan terhadap salah satu data contohnya dari 2068 menjadi 1968
loan_data['earliest_cr_line_date'][628]


# In[37]:


# Mengubah dtype menjadi datetime lagi
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line_date'])
loan_data['earliest_cr_line_date']


# In[38]:


# Pengecekan kembali data untuk melihat perubahaanya (asumsi kan pada bulan desember 2015)
loan_data['mths_since_earliest_cr_line_date'] = round(pd.to_numeric((pd.to_datetime('2015-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
# Menampilkan statistik deskriptif dari kolom ini
loan_data['mths_since_earliest_cr_line_date'].describe()


# **Catatan**: Sudah tidak ada nilai negatif lagi dan data sudah dikonversi dengan benar.

# In[39]:


# drop column earliest_cr_line_date, mths_since_earliest_cr_line, and earliest_cr_line sebagaimana kita tidak membutuhkan data nya lagi.
loan_data.drop(columns = ['earliest_cr_line_date' ,'mths_since_earliest_cr_line', 
                          'earliest_cr_line'], inplace = True)


# #### 4. last_credit_pull_d
# kapan hari terakhir LC menge'check' credit history

# In[40]:


loan_data['last_credit_pull_d']


# In[41]:


# Asusmsikan sekarang adalah bulan Desember 2017
# Ekstrak tanggal dan waktu dari variabel string yang ada dalam format tertentu. dan isi data NaN dengan max date
loan_data['last_credit_pull_d'] = pd.to_datetime(loan_data['last_credit_pull_d'], format = '%b-%y').fillna(pd.to_datetime("2016-01-01"))

# hitung selisih antara dua tanggal dalam bulan, ubah menjadi tipe data numerik dan bulatkan.
loan_data['mths_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['last_credit_pull_d']) / np.timedelta64(1, 'M')))

# Menampilkan beberapa statistik deskriptif untuk nilai kolom.
loan_data['mths_since_last_credit_pull_d'].describe()


# **Catatan**: Sudah tidak ada nilai negatif lagi dan data sudah dikonversi dengan benar.

# In[42]:


#drop column last_credit_pull_d sebagaimana kita tidak membutuhkan data nya lagi.
loan_data.drop(columns = ['last_credit_pull_d'], inplace = True)


# ### Checking for missing value (again)

# In[43]:


#Checking for missing values
loan_data.isnull().sum()


# In[44]:


# drop all rows that contain a lot of missing value
loan_data.dropna(subset = ['revol_util'], inplace = True)

#reset index
loan_data.reset_index(drop= True, inplace = True)


# **Catatan**: Sudah tidak ada missing value di semua kolom.

# ### Explore Data

# In[45]:


def risk_percentage(x):
    ratio = (loan_data.groupby(x)['good_bad_loan'] # group by
         .value_counts(normalize=True) # calculate the ratio
         .mul(100) # multiply by 100 to be percent
         .rename('risky (%)') # rename column as percent
         .reset_index())

    sns.lineplot(data=ratio[ratio['good_bad_loan'] == 0], x=x, y='risky (%)')
    plt.title(x)
    plt.show()


# In[46]:


print(loan_data.nunique()[loan_data.nunique() < 12].sort_values().index)


# In[47]:


#unique columns and months date column
unq_cols = ['term', 'initial_list_status', 'verification_status',
       'home_ownership', 'acc_now_delinq', 'grade', 'inq_last_6mths',
       'collections_12_mths_ex_med', 'emp_length', 'mths_since_earliest_cr_line_date', 'mths_since_last_credit_pull_d']
for cols in unq_cols:
    risk_percentage(cols)


# **Insight**:
# - term: memiliki risiko rendah pada term 36 dan risiko tinggi pada term 60.
# - initial_list_status: memiliki risiko tinggi pada f dan risiko rendah pada w
# - verification_status: memiliki risiko tinggi status diverifikasi
# - home_ownership: memiliki risiko tinggi pada jenis kepemilikan None dan Other.
# - acc_now_delinq: memiliki risiko rendah 2, 0, 1, dan memiliki nilai risiko tinggi 3, 4, 5
# - grade: ada peningkatan risiko yang terkait dengan ini.
# - inq_last_6mths: ada peningkatan risiko yang terkait dengan ini.
# - collections_12_mths_ex_med: memiliki risiko rendah dengan nilai 3,0 dan memiliki risiko tinggi dengan nilai 4,0
# - Employment length: masa kerja kurang dari 1 tahun memiliki persentase risiko terbesar dan masa kerja lebih dari 9 tahun memiliki persentase risiko terkecil.
# - Months since earliest cr line date: Semakin awal batas kredit, semakin stabil catatan peminjam, dan ada peningkatan risiko yang terkait dengannya.
# - months since last credit pull date: memiliki variasi persentase risiko yang berbeda.

# In[48]:


#Check correlation
plt.figure(figsize=(20,20))
sns.heatmap(loan_data.corr(), annot=True, annot_kws={'size':14})


# ### One Hot Encoding
# 
# Mengonversikan kolom kategorikal dengan One Hot Encoding.

# In[49]:


# Convert categorical columns with One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_cols = [col for col in loan_data.select_dtypes(include='object').columns.tolist()]
onehot_cols = pd.get_dummies(loan_data[cat_cols], drop_first=True)


# In[50]:


onehot_cols


# ### Standardization
# 
# Semua kolom numerik distandarisasi dengan StandardScaler.

# In[52]:


from sklearn.preprocessing import StandardScaler

num_cols = [col for col in loan_data.columns.tolist() if col not in cat_cols + ['good_bad_loan']]
ss = StandardScaler()
std_cols = pd.DataFrame(ss.fit_transform(loan_data[num_cols]), columns=num_cols)


# In[53]:


std_cols


# ## Get Final Data

# In[54]:


# combining column
final_data = pd.concat([onehot_cols, std_cols, loan_data[['good_bad_loan']]], axis=1)
final_data.head()


# ## Model Training and Prediction (Main Process)

# ### Data Splitting
# 
# Membuat dataset training dan dataset test, dengan perbandingan 80% untuk data training dan 20% untuk data testing

# In[55]:


# separate dependant (y) and independant (X) variable
X = final_data.drop('good_bad_loan', axis = 1)
y = final_data['good_bad_loan']


# In[56]:


#spliting data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42,stratify=y)


# In[57]:


X_train.shape, X_test.shape


# ### Checking for Class Imbalance in Final Dataset

# In[58]:


#check if class labels are balanced
plt.title('Good (1) vs Bad (0) Loans Balance')
sns.barplot(x=final_data.good_bad_loan.value_counts().index,y=final_data.good_bad_loan.value_counts().values)


# In[59]:


#checking  imbalance for training dataset
y_train.value_counts()


# Dari grafik dan data di atas, data yang tergolong kredit macet memiliki data yang lebih sedikit dibandingkan dengan data kredit bagus. Dengan demikian, dataset ini memiliki data yang **tidak seimbang.**

# ### Train the model without handling the imbalanced class distribution

# In[61]:


# Import Library
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve


# Melatih model tanpa menangani distribusi kelas yang tidak seimbang menggunakan **Logistic Regression.**

# In[62]:


# training
LR= LogisticRegression(max_iter=600).fit(X_train, y_train)
# predicting
y_pred_LR = LR.predict(X_test)

# classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_LR, digits=4, target_names = target_names))


# #### Catatan:
# Hasil prediksi memiliki hasil yang sangat tidak seimbang antara kelas pinjaman yang buruk dan pinjaman yang baik. dimana untuk kelas kredit macet, hasil recall yang didapatkan mendekati nol. sedangkan pada kelas pinjaman yang baik, nilai recall yang didapat hampir 100%. **Recall** adalah jumlah "positif" yang diprediksi dengan benar dibagi dengan jumlah total "positif". Itu berarti model mengidentifikasi dengan benar 2,80% dari total kredit macet dan mengidentifikasi dengan benar 99,56% dari total kredit bagus.
# 
# **Model memperoleh akurasi yang cukup tinggi hanya dengan memprediksi kelas mayoritas, tetapi gagal menangkap kelas minoritas,**
# 
# Hal ini mungkin disebabkan oleh dataset yang tidak seimbang sehingga model machine learning mengabaikan kelas minoritas (bad loan class) seluruhnya.
# 
# Jadi, ketidakseimbangan kelas ini dapat mempengaruhi model pada pelatihan. Hal ini menjadi ***masalah*** karena diperlukan data kredit macet (kelas minoritas) untuk model prediksi ini
# 
# Teknik **OVERSAMPLING** pada kelas minoritas akan digunakan untuk mengatasi ketidakseimbangan data ini.

# In[ ]:





# ### Oversampling Minority Class to Resolve Class Imbalance
# 
# Random Oversampling melibatkan duplikasi contoh secara acak dari kelas minoritas dan menambahkannya ke dataset pelatihan.

# In[64]:


get_ipython().system('pip install imbalanced-learn')


# In[65]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

#check value counts before and after oversampling
print('Before OverSampling:\n{}'.format(y_train.value_counts()))
print('\nAfter OverSampling:\n{}'.format(y_train_ros.value_counts()))


# ## Train the model after over sampling

# In[67]:


get_ipython().system('pip install xgboost')


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ### 1. Logistic Regression

# In[69]:


# Training 
LR_ros= LogisticRegression(max_iter=600)  
LR_ros.fit(X_train_ros, y_train_ros)

#predicting
y_pred_LR_ros = LR_ros.predict(X_test)

#classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_LR_ros, digits=4, target_names = target_names))


# ### 2. Random Forest

# In[70]:


#building model
rf_ros = RandomForestClassifier(max_depth=10, n_estimators=20)
rf_ros.fit(X_train_ros, y_train_ros)

#predicting
y_pred_rf_ros = rf_ros.predict(X_test)

#classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_rf_ros, digits=4, target_names = target_names))


# ### 3. Decision Tree

# In[71]:


#building model
dt_ros = DecisionTreeClassifier(max_depth = 10)
dt_ros.fit(X_train_ros, y_train_ros)

#predicting
y_pred_dt_ros = dt_ros.predict(X_test)

#classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_dt_ros, digits=4, target_names = target_names))


# ### 4. XGBOOST

# In[73]:


#building model
from xgboost import XGBClassifier
xgb_ros = XGBClassifier(max_depth=5)
xgb_ros.fit(X_train_ros, y_train_ros)

#predicting
y_pred_xgb_ros = xgb_ros.predict(X_test)

#classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_xgb_ros, digits=4, target_names = target_names))


# ### 5. AdaBoost

# In[74]:


#building model
adb_ros = AdaBoostClassifier(n_estimators = 100)
adb_ros.fit(X_train_ros, y_train_ros)

#predicting
y_pred_adb_ros = adb_ros.predict(X_test)

#classification report
target_names = ['bad loan', 'good loan']
print('Classification_Report:')
print(classification_report(y_test, y_pred_adb_ros, digits=4, target_names = target_names))


# ## Kesimpulan

# - Setelah dilakukan model pelatihan dengan data oversampling, diperoleh akurasi tiap kelas (bad loan dan good loan) dengan nilai yang cukup stabil (rata-rata akurasi tiap kelas > 60%). Sehingga dapat dikatakan bahwa penggunaan oversampling dapat membantu model pada saat pelatihan sehingga dapat mendeteksi kelas pinjaman yang buruk dan pinjaman yang baik dengan cukup baik.
# 
# - Rata-rata hasil akurasi terbaik diantara semua model di atas adalah menggunakan ***XGBoost Classifier*** dengan rata-rata nilai akurasi sebesar 71,06% (bad loan recall = 63,3% dan good loan recall = 71,96%). Walaupun nilai akurasi ini masih belum tinggi, namun nilai ini sudah cukup tinggi karena dataset yang tidak seimbang. **Recall** adalah jumlah "positif" yang diprediksi dengan benar dibagi dengan jumlah total "positif". Artinya, model ini mengidentifikasi dengan benar 63,3% dari total kredit macet dan mengidentifikasi dengan benar 71,96% dari total kredit bagus.

# In[ ]:





# In[ ]:





# In[ ]:




