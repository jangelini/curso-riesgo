# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:47:08 2022

@author: Nora
"""

#Clase I

import os

import pandas as pd
import numpy as np

DATA_PATH = 'data'

appl = pd.read_csv(os.path.join(DATA_PATH, 'application_train.csv'))

bureau =  pd.read_csv('C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/bureau.csv')

bureau_balance = pd.read_csv('C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/bureau_balance.csv')

card_balance =  pd.read_csv( 'C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/credit_card_balance.csv')

installments = pd.read_csv ('C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/installments_payments.csv')

prev =  pd.read_csv('C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/previous_application.csv')

pos_cash = pd.read_csv('C:/Users/Nora/Desktop/curso_untref/curso_riesgo2/home-credit-default-risk/POS_CASH_balance.csv')


#para guardar un dataframe

tabla.to_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/datos_a1.pkl")

tabla = pd.read_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/datos_a1.pkl")


################

col1=appl.columns

yy=appl.describe()



z=0
##estadistic
z2=appl #dataframe
columns=(z2.columns)
z1=len(columns)
yy=z2.describe() #estadisticas de las variables numericas

z=0
p1=pd.DataFrame(z2[columns[z]].value_counts(normalize=True)).count()
while z< z1:
    p2=pd.DataFrame(z2[columns[z]].value_counts(normalize=True)).count()
    p1=p1.append(p2)
    z=z+1

p1=p1.reset_index()


p1_d1=p1.loc[(p1[0])<20] #nos quedamos con las que tienen mens de 20 valores

p2=p1_d1['index'].tolist()

p3=z2[p2] #hacemos un dataframe solo con esas variables


yy1=p3[p2[0]].value_counts()
yy1['name']=p2[0]
z1=len(p2)
z=0
while z<z1:
    yy2=p3[p2[z]].value_counts()
    yy2['name']=p2[z]
    yy1=yy1.append(yy2)
    z=z+1

    
#Los resultados en yy y en yy1


######################################################
    

###################CLASE II ###################################

#Bureau



datos_a1=bureau.loc[(bureau.index<40)]



##KEY=SK_ID_CURR

bureau_consumer=bureau.loc[(bureau['CREDIT_TYPE']=='Consumer credit')]
bureau_card=bureau.loc[(bureau['CREDIT_TYPE']=='Credit card')]
bureau_car=bureau.loc[(bureau['CREDIT_TYPE']=='Car loan')]
bureau_mort=bureau.loc[(bureau['CREDIT_TYPE']=='Mortgage')]

##############################################################################

bureau_consumer['one']=1

bureau_consumer_k=bureau_consumer[["SK_ID_CURR","CREDIT_DAY_OVERDUE",
                                  "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                                  "one"]].iloc[:,:]

#cambuamos los nombres de las columnas

bureau_consumer_k.columns=["SK_ID_CURR","CREDIT_DAY_OVERDUE_co",
                                  "AMT_CREDIT_MAX_OVERDUE_co", "AMT_CREDIT_SUM_co", "AMT_CREDIT_SUM_DEBT_co",
                                  "one_co"]

bureau_consumer_sum=bureau_consumer_k.groupby(["SK_ID_CURR"]).sum()
bureau_consumer_max=bureau_consumer_k.groupby(["SK_ID_CURR"]).max()

####################################################################################


bureau_car['one']=1

bureau_car_k=bureau_car[["SK_ID_CURR","CREDIT_DAY_OVERDUE",
                                  "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                                  "one"]].iloc[:,:]

#cambuamos los nombres de las columnas

bureau_car_k.columns=["SK_ID_CURR","CREDIT_DAY_OVERDUE_ca",
                                  "AMT_CREDIT_MAX_OVERDUE_ca", "AMT_CREDIT_SUM_ca", "AMT_CREDIT_SUM_DEBT_ca",
                                  "one_ca"]

bureau_car_sum=bureau_car_k.groupby(["SK_ID_CURR"]).sum()
bureau_car_max=bureau_car_k.groupby(["SK_ID_CURR"]).max()

####################################################################################


bureau_mort['one']=1

bureau_mort_k=bureau_mort[["SK_ID_CURR","CREDIT_DAY_OVERDUE",
                                  "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                                  "one"]].iloc[:,:]

#cambuamos los nombres de las columnas

bureau_mort_k.columns=["SK_ID_CURR","CREDIT_DAY_OVERDUE_mo",
                                  "AMT_CREDIT_MAX_OVERDUE_mo", "AMT_CREDIT_SUM_mo", "AMT_CREDIT_SUM_DEBT_mo",
                                  "one_mo"]

bureau_mort_sum=bureau_mort_k.groupby(["SK_ID_CURR"]).sum()
bureau_mort_max=bureau_mort_k.groupby(["SK_ID_CURR"]).max()


#####################################################################################


bureau_card['one']=1

bureau_card_k=bureau_card[["SK_ID_CURR","CREDIT_DAY_OVERDUE",
                                  "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                                  "one"]].iloc[:,:]

#cambuamos los nombres de las columnas

bureau_card_k.columns=["SK_ID_CURR","CREDIT_DAY_OVERDUE_cd",
                                  "AMT_CREDIT_MAX_OVERDUE_cd", "AMT_CREDIT_SUM_cd", "AMT_CREDIT_SUM_DEBT_cd",
                                  "one_cd"]

bureau_card_sum=bureau_card_k.groupby(["SK_ID_CURR"]).sum()
bureau_card_max=bureau_card_k.groupby(["SK_ID_CURR"]).max()

#####################################################################################

appl1=appl.merge(bureau_consumer_sum, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl2=appl1.merge(bureau_car_sum, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl3=appl2.merge(bureau_mort_sum, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl4=appl3.merge(bureau_card_sum, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")

appl5=appl4.merge(bureau_consumer_max, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl6=appl5.merge(bureau_car_max, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl7=appl6.merge(bureau_mort_max, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")
appl8=appl7.merge(bureau_card_max, how='left', left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")


###################################################################################################
###Bureau Balance


datos_a2=bureau_balance.loc[(bureau_balance.index<40)]

bureau_balance['one']=1


bureau_balance_gr=bureau_balance.groupby(["STATUS"]).sum()


bureau_balance['STATUS_2']=np.where(bureau_balance['STATUS']=="0", 0,
                                    np.where(bureau_balance['STATUS']=="1", 1,
                                    np.where(bureau_balance['STATUS']=="2", 2,
                                    np.where(bureau_balance['STATUS']=="3", 3,                                             
                                    np.where(bureau_balance['STATUS']=="4", 4,                                             
                                    np.where(bureau_balance['STATUS']=="5", 5,                                             
                                    np.where(bureau_balance['STATUS']=="C", 6,                                             
                                    np.where(bureau_balance['STATUS']=="X", 7, 99
                                             )))))))).astype(int)



bureau_balance   = pd.get_dummies(bureau_balance, columns=['STATUS'], prefix_sep="_*_")

bureau_balance_1 = bureau_balance.drop(['MONTHS_BALANCE'], axis=1)


bureau_merge     = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
bureau_balance_2 = bureau_balance_1.merge(bureau_merge, how='left', left_on=['SK_ID_BUREAU'], right_on=['SK_ID_BUREAU'], sort="True")
bureau_balance_3 = bureau_balance_2.drop(['SK_ID_BUREAU'], axis=1)
bureau_balance_4 = bureau_balance_3.groupby(["SK_ID_CURR"]).sum()
bureau_balance_5 = bureau_balance_4.reset_index()

appl9=appl8.merge(bureau_balance_5, how="left", left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")


#########################################################################################

###PREV


datos_a3=prev.loc[(prev.index<40)]


prev['one']=1

prev1=prev[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT','one',"SK_ID_CURR"]]

prev2=prev1.groupby(["SK_ID_CURR"]).sum()

appl10=appl9.merge(prev2, how="left", left_on=["SK_ID_CURR"], right_on=["SK_ID_CURR"], sort="True")



#para guardar un dataframe

appl10.to_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/appl10.pkl")

appl110a = pd.read_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/appl10.pkl")



#CLASE III
#####################################################
##Card Balance

card_balance1=card_balance.groupby(['SK_ID_CURR']).agg({'AMT_BALANCE':['mean','max'], 'AMT_CREDIT_LIMIT_ACTUAL':['mean','max'],
                                                        'AMT_DRAWINGS_ATM_CURRENT':['mean','max'],
                                                        'AMT_DRAWINGS_CURRENT':['mean','max'],
                                                        'AMT_DRAWINGS_POS_CURRENT':['mean','max'],
                                                        'AMT_PAYMENT_TOTAL_CURRENT':['mean','max'],
                                                        'AMT_TOTAL_RECEIVABLE':['mean','max'],
                                                        'SK_DPD':['mean','max']})


datos_a1=card_balance1.loc[(card_balance1.index<100047)]


card_balance2=card_balance1.reset_index()

card_balance2.columns=['SK_ID_CURR','AMT_BALANCE_me','AMT_BALANCE_ma', 
                       'AMT_CREDIT_LIMIT_ACTUAL_me','AMT_CREDIT_LIMIT_ACTUAL_ma',
                       'AMT_DRAWINGS_ATM_CURRENT_me','AMT_DRAWINGS_ATM_CURRENT_ma',
                       'AMT_DRAWINGS_CURRENT_me', 'AMT_DRAWINGS_CURRENT_ma',
                       'AMT_DRAWINGS_POS_CURRENT_me', 'AMT_DRAWINGS_POS_CURRENT_ma',
                       'AMT_PAYMENT_TOTAL_CURRENT_me', 'AMT_PAYMENT_TOTAL_CURRENT_ma',
                       'AMT_TOTAL_RECEIVABLE_me', 'AMT_TOTAL_RECEIVABLE_ma',
                       'SK_DPD_me','SK_DPD_ma']

datos_a2=card_balance2.loc[(card_balance1.index<100047)]


appl11=appl10.merge(card_balance2, how='left', left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], sort='True')


###############################################################################
##POS CASH

pos_cash1=pos_cash.drop(['NAME_CONTRACT_STATUS', 'SK_ID_PREV'], axis=1)

pos_cash2=pos_cash1.groupby(['SK_ID_CURR']).mean()

appl12=appl11.merge(pos_cash2, how='left', left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], sort='True')


#############################################################################

#Installments

instalments1=installments.drop(['SK_ID_PREV'], axis=1)

installments2=instalments1.groupby(['SK_ID_CURR']).mean()

appl13=appl12.merge(installments2, how='left', left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], sort='True')


###########################################################################

appl14=appl13.drop(['NAME_CONTRACT_TYPE','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',	
'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'], axis=1)

appl14.fillna(0, inplace=True)


appl14.to_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/appl14.pkl")

appl14 = pd.read_pickle("C:/Users/Nora/Desktop/curso_untref/curso_riesgo2022/appl14.pkl")

##################################################################################

from sklearn.model_selection import train_test_split

x=appl14

x=x.drop(['TARGET'], axis=1)

y=appl14['TARGET']

y.value_counts()

x_norm=(x-x.mean())/x.std()

x_train, x_valid, y_train, y_valid = train_test_split (x_norm, y, test_size=0.2)

##################################################################################


y.value_counts(normalize=True)
y_train.value_counts(normalize=True)
y_valid.value_counts(normalize=True)

####LOGISTICA###############

from sklearn.linear_model import LogisticRegression

# https://stackoverflow.com/questions/22851316/what-is-the-inverse-of-regularization-strength-in-logistic-regression-how-shoul
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html




model1 = LogisticRegression(solver='liblinear', random_state=0, verbose=3, C=1.0, penalty='l2')
model1.fit(x_train, y_train)




coef1=pd.DataFrame(model1.coef_).T

coef1['col']=pd.DataFrame(x_train.columns)