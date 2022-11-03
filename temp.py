import pandas as pd
import numpy as np
import sys
import sklearn
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import os
#from google.colab import drive
from sklearn.preprocessing import LabelEncoder
def onelineencode():
  protocol_type_le = LabelEncoder()
  service_le = LabelEncoder()
  flag_le = LabelEncoder()
  data_Train['protocol_type'] = protocol_type_le.fit_transform(data_Train['protocol_type'])
  data_Train['service'] = service_le.fit_transform(data_Train['service'])
  data_Train['flag'] = flag_le.fit_transform(data_Train['flag'])
  data_Test['protocol_type'] = protocol_type_le.fit_transform(data_Test['protocol_type'])
  data_Test['service'] = service_le.fit_transform(data_Test['service'])
  data_Test['flag'] = flag_le.fit_transform(data_Test['flag'])
  data_Validate['protocol_type'] = protocol_type_le.fit_transform(data_Validate['protocol_type'])
  data_Validate['service'] = service_le.fit_transform(data_Validate['service'])
  data_Validate['flag'] = flag_le.fit_transform(data_Validate['flag'])
def multi_preprocess():
  l1=['normal','dos','probe','u2r','r2l']
  tdf=df_train['attack']
  nwdf=tdf.replace({ 'normal' : l1[0], 'neptune' : l1[1] ,'back': l1[1], 'land': l1[1], 'pod': l1[1], 'smurf': l1[1], 'teardrop': l1[1],'mailbomb': l1[1], 'apacl1[1]he2': l1[1], 'processtable': l1[1], 'udpstorm': l1[1], 'worm': l1[1],
                           'ipsweep' : l1[2],'nmap' : l1[2],'portsweep' : l1[2],'satan' : l1[2],'mscan' : l1[2],'saint' : l1[2]
                           ,'ftp_write': l1[3],'guess_passwd':l1[3],'imap': l1[3],'multihop': l1[3],'phf': l1[3],'spy': l1[3],'warezclient': l1[3],'warezmaster': l1[3],'sendmail': l1[3],'named': l1[3],'snmpgetattack': l1[3],'snmpguess': l1[3],'xlock': l1[3],'xsnoop': l1[3],'httptunnel': l1[3],
                           'buffer_overflow': l1[4],'loadmodule': l1[4],'perl': l1[4],'rootkit': l1[4],'ps': l1[4],'sqlattack': l1[4],'xterm': l1[4],
                 'mscan':l1[2], 'udpstorm':l1[1], 'xterm':l1[3], 'worm':l1[1], 'saint':l1[2], 'snmpgetattack':l1[4], 'named':l1[4], 'mailbomb':l1[4], 'apache2':l1[1], 'httptunnel':l1[4], 'ps':l1[3], 'xsnoop':l1[4], 'processtable':l1[1], 'sendmail':l1[4], 'snmpguess':l1[4], 'sqlattack':l1[3], 'xlock':l1[4]})
  df_train['attack']=nwdf
  l1=['normal','dos','probe','u2r','r2l']
  tdf=df_test['attack']
  nwdf=tdf.replace({ 'normal' : l1[0], 'neptune' : l1[1] ,'back': l1[1], 'land': l1[1], 'pod': l1[1], 'smurf': l1[1], 'teardrop': l1[1],'mailbomb': l1[1], 'apacl1[1]he2': l1[1], 'processtable': l1[1], 'udpstorm': l1[1], 'worm': l1[1],
                           'ipsweep' : l1[2],'nmap' : l1[2],'portsweep' : l1[2],'satan' : l1[2],'mscan' : l1[2],'saint' : l1[2]
                           ,'ftp_write': l1[3],'guess_passwd':l1[3],'imap': l1[3],'multihop': l1[3],'phf': l1[3],'spy': l1[3],'warezclient': l1[3],'warezmaster': l1[3],'sendmail': l1[3],'named': l1[3],'snmpgetattack': l1[3],'snmpguess': 3,'xlock': l1[3],'xsnoop': l1[3],'httptunnel': l1[3],
                           'buffer_overflow': l1[4],'loadmodule': l1[4],'perl': l1[4],'rootkit': l1[4],'ps': l1[4],'sqlattack': l1[4],'xterm': l1[4],
                 'mscan':l1[2], 'udpstorm':l1[1], 'xterm':l1[3], 'worm':l1[1], 'saint':l1[2], 'snmpgetattack':l1[4], 'named':l1[4], 'mailbomb':l1[4], 'apache2':l1[1], 'httptunnel':l1[4], 'ps':l1[3], 'xsnoop':l1[4], 'processtable':l1[1], 'sendmail':l1[4], 'snmpguess':l1[4], 'sqlattack':l1[3], 'xlock':l1[4]})
  df_test['attack']=nwdf

  l1=['normal','dos','probe','u2r','r2l']
  tdf=df_validate['attack']
  nwdf=tdf.replace({ 'normal' : l1[0], 'neptune' : l1[1] ,'back': l1[1], 'land': l1[1], 'pod': l1[1], 'smurf': l1[1], 'teardrop': l1[1],'mailbomb': l1[1], 'apacl1[1]he2': l1[1], 'processtable': l1[1], 'udpstorm': l1[1], 'worm': l1[1],
                           'ipsweep' : l1[2],'nmap' : l1[2],'portsweep' : l1[2],'satan' : l1[2],'mscan' : l1[2],'saint' : l1[2]
                           ,'ftp_write': l1[3],'guess_passwd':l1[3],'imap': l1[3],'multihop': l1[3],'phf': l1[3],'spy': l1[3],'warezclient': l1[3],'warezmaster': l1[3],'sendmail': l1[3],'named': l1[3],'snmpgetattack': l1[3],'snmpguess': 3,'xlock': l1[3],'xsnoop': l1[3],'httptunnel': l1[3],
                           'buffer_overflow': l1[4],'loadmodule': l1[4],'perl': l1[4],'rootkit': l1[4],'ps': l1[4],'sqlattack': l1[4],'xterm': l1[4],
                 'mscan':l1[2], 'udpstorm':l1[1], 'xterm':l1[3], 'worm':l1[1], 'saint':l1[2], 'snmpgetattack':l1[4], 'named':l1[4], 'mailbomb':l1[4], 'apache2':l1[1], 'httptunnel':l1[4], 'ps':l1[3], 'xsnoop':l1[4], 'processtable':l1[1], 'sendmail':l1[4], 'snmpguess':l1[4], 'sqlattack':l1[3], 'xlock':l1[4]})
  df_validate['attack']=nwdf
def multiclassaccuracy():
  classifier = KNeighborsClassifier(n_neighbors=3)
  classifier.fit(x_train, y_train)
  x_predict=classifier.predict(x_test)
  print('Multi class Accuracy: ',accuracy_score(x_predict,y_test))

def binarypreprocess():
  l1=[1,0,0,0,0]
  tdf=y_bin_train['attack']
  nwdf=tdf.replace({ 'normal' : l1[0], 'neptune' : l1[1] ,'back': l1[1], 'land': l1[1], 'pod': l1[1], 'smurf': l1[1], 'teardrop': l1[1],'mailbomb': l1[1], 'apacl1[1]he2': l1[1], 'processtable': l1[1], 'udpstorm': l1[1], 'worm': l1[1],
                           'ipsweep' : l1[2],'nmap' : l1[2],'portsweep' : l1[2],'satan' : l1[2],'mscan' : l1[2],'saint' : l1[2]
                           ,'ftp_write': l1[3],'guess_passwd':l1[3],'imap': l1[3],'multihop': l1[3],'phf': l1[3],'spy': l1[3],'warezclient': l1[3],'warezmaster': l1[3],'sendmail': l1[3],'named': l1[3],'snmpgetattack': l1[3],'snmpguess': l1[3],'xlock': l1[3],'xsnoop': l1[3],'httptunnel': l1[3],
                           'buffer_overflow': l1[4],'loadmodule': l1[4],'perl': l1[4],'rootkit': l1[4],'ps': l1[4],'sqlattack': l1[4],'xterm': l1[4],
                 'mscan':l1[2], 'udpstorm':l1[1], 'xterm':l1[3], 'worm':l1[1], 'saint':l1[2], 'snmpgetattack':l1[4], 'named':l1[4], 'mailbomb':l1[4], 'apache2':l1[1], 'httptunnel':l1[4], 'ps':l1[3], 'xsnoop':l1[4], 'processtable':l1[1], 'sendmail':l1[4], 'snmpguess':l1[4], 'sqlattack':l1[3], 'xlock':l1[4]})
  y_bin_train['attack']=nwdf
  l1=[1,0,0,0,0]
  tdf=y_bin_test['attack']
  nwdf=tdf.replace({ 'normal' : l1[0], 'neptune' : l1[1] ,'back': l1[1], 'land': l1[1], 'pod': l1[1], 'smurf': l1[1], 'teardrop': l1[1],'mailbomb': l1[1], 'apacl1[1]he2': l1[1], 'processtable': l1[1], 'udpstorm': l1[1], 'worm': l1[1],
                           'ipsweep' : l1[2],'nmap' : l1[2],'portsweep' : l1[2],'satan' : l1[2],'mscan' : l1[2],'saint' : l1[2]
                           ,'ftp_write': l1[3],'guess_passwd':l1[3],'imap': l1[3],'multihop': l1[3],'phf': l1[3],'spy': l1[3],'warezclient': l1[3],'warezmaster': l1[3],'sendmail': l1[3],'named': l1[3],'snmpgetattack': l1[3],'snmpguess': l1[3],'xlock': l1[3],'xsnoop': l1[3],'httptunnel': l1[3],
                           'buffer_overflow': l1[4],'loadmodule': l1[4],'perl': l1[4],'rootkit': l1[4],'ps': l1[4],'sqlattack': l1[4],'xterm': l1[4],
                 'mscan':l1[2], 'udpstorm':l1[1], 'xterm':l1[3], 'worm':l1[1], 'saint':l1[2], 'snmpgetattack':l1[4], 'named':l1[4], 'mailbomb':l1[4], 'apache2':l1[1], 'httptunnel':l1[4], 'ps':l1[3], 'xsnoop':l1[4], 'processtable':l1[1], 'sendmail':l1[4], 'snmpguess':l1[4], 'sqlattack':l1[3], 'xlock':l1[4]})
  y_bin_test['attack']=nwdf

def binaryclassaccuracy():
  classifier = KNeighborsClassifier(n_neighbors=3)
  classifier.fit(x_bin_train, y_bin_train)
  x_bin_predict=classifier.predict(x_bin_test)
  print('binary class accuracy= ',accuracy_score(x_bin_predict,y_bin_test))
def advance():
  classifier = KNeighborsClassifier(n_neighbors=3)
  classifier.fit(x_bin_train, y_bin_train)
  tp=x_validate.sample()
  val=classifier.predict(tp)
  if(val==1):
      print('Binary class Type: NORMAL')
      classifier2 = KNeighborsClassifier(n_neighbors=3)
      classifier2.fit(x_train, y_train)
      temp=classifier2.predict(tp)
      print('type of attack:',temp)
      if(temp=='normal'):
        print('this is safe.')
  elif(val==0):
      print('Binary class Type: ATTACK')
      classifier2 = KNeighborsClassifier(n_neighbors=3)
      classifier2.fit(x_train, y_train)
      temp=classifier2.predict(tp)
      print('type of attack',temp)
      if(temp=='dos'):
        print('A Denial-of-Service (DoS) attack is an attack meant to shut down a machine or network, making it inaccessible to its intended users. DoS attacks accomplish this by flooding the target with traffic, or sending it information that triggers a crash. In both instances, the DoS attack deprives legitimate users (i.e. employees, members, or account holders) of the service or resource they expected.')
      elif(temp=='probe'):
        print('Probing is another type of attack in which the intruder scans network devices to determine weakness in topology design or some opened ports and then use them in the future for illegal access to personal information.')
      elif(temp=='r2l'):
        print('Remote to user (R2L) is a type of computer network attacks, in which an intruder sends set of packets to another computer or server over a network where he/she does not have permission to access as a local user.')
      elif(temp=='u2r'):
        print('User to root attacks (U2R) is an another type of attack where the intruder tries to access the network resources as a normal user,  and after several attempts, the intruder becomes as a full access user.')
data_Train =pd.read_csv('KDDTrain+.csv')
data_Test = pd.read_csv('KDDTest+.csv')
data_Validate=pd.read_csv('new validation project.csv')
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent'
            ,'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root'
            ,'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
            ,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate'
            ,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
            ,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'
            ,'dst_host_srv_rerror_rate','attack'])
data_Train.columns=columns
data_Test.columns=columns
data_Validate.columns=columns
onelineencode()
df_train=data_Train.copy(deep=True)
df_test=data_Test.copy(deep=True)
df_validate=data_Validate.copy(deep=True)
multi_preprocess()
x_train=df_train.drop(['attack'],axis=1)
y_train=pd.DataFrame(df_train['attack'].copy())
x_test=df_test.drop(['attack'],axis=1)
y_test=pd.DataFrame(df_test['attack'].copy())
x_validate=df_validate.drop(['attack'],axis=1)
y_validate=pd.DataFrame(df_validate['attack'].copy())

label_encoder = LabelEncoder() 
scaler=MinMaxScaler()
x1=x_train.copy(deep=True)
scaler=MinMaxScaler()
scaler.fit(x1)
scaled_data=scaler.transform(x1)
scaled_data=pd.DataFrame(scaled_data)
scaled_data.columns= x1.columns
x_train=scaled_data

label_encoder = LabelEncoder() 
scaler=MinMaxScaler()
x1=x_test.copy(deep=True)
scaler=MinMaxScaler()
scaler.fit(x1)
scaled_data=scaler.transform(x1)
scaled_data=pd.DataFrame(scaled_data)
scaled_data.columns= x1.columns
x_test=scaled_data

label_encoder = LabelEncoder() 
scaler=MinMaxScaler()
x1=x_validate.copy(deep=True)
scaler=MinMaxScaler()
scaler.fit(x1)
scaled_data=scaler.transform(x1)
scaled_data=pd.DataFrame(scaled_data)
scaled_data.columns= x1.columns
x_validate=scaled_data
x_bin_train=data_Train.drop(['attack'],axis=1)
y_bin_train=pd.DataFrame(data_Train['attack']).copy()
x_bin_test=data_Test.drop(['attack'],axis=1)
y_bin_test=pd.DataFrame(data_Test['attack']).copy()
binarypreprocess()
label_encoder = LabelEncoder() 
scaler=MinMaxScaler()
x1=x_bin_train.copy(deep=True)
scaler=MinMaxScaler()
scaler.fit(x1)
scaled_data=scaler.transform(x1)
scaled_data=pd.DataFrame(scaled_data)
scaled_data.columns= x1.columns
x_bin_train=scaled_data
label_encoder = LabelEncoder() 
scaler=MinMaxScaler()
x1=x_bin_test.copy(deep=True)
scaler=MinMaxScaler()
scaler.fit(x1)
scaled_data=scaler.transform(x1)
scaled_data=pd.DataFrame(scaled_data)
scaled_data.columns= x1.columns
x_bin_test=scaled_data
l=[]
for i in y_test.values:
  for j in i:
    l.append(j)
y_test=np.array(l)
l=[]
for i in y_train.values:
  for j in i:
    l.append(j)
y_train=np.array(l)
l=[]
for i in y_bin_train.values:
  for j in i:
    l.append(j)
y_bin_train=np.array(l)
l=[]
for i in y_bin_test.values:
  for j in i:
    l.append(j)
y_bin_test=np.array(l)
advance()
#binaryclassaccuracy()
#multiclassaccuracy()
