#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:28:02 2017

@author: hhtseng
"""

### basic loading 
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from numpy import genfromtxt
import os
from sklearn.decomposition import PCA
import pylab
from scipy import signal

sp500 = os.listdir("daily")
num=len(sp500)
choose_num=1825
sp_price_daily=[]
sp_volume_daily=[]
sp_pm_ratio_daily=[]
name_include=[]


#%% reading sp500 based on choose_num 
for i in range(500):
    choose= genfromtxt('./daily/' + sp500[i], delimiter=',') # 2:open 3:max 4:min 5:close 6:volume  
    if choose.shape[0]<choose_num:
        print(sp500[i])                   
    else:
        sp_price_daily.append((choose[-choose_num:,3]+choose[-choose_num:,4])/2)
        sp_volume_daily.append(choose[-choose_num:,6])
        sp_pm_ratio_daily.append((choose[-choose_num:,5]-choose[-choose_num:,2])/choose[-choose_num:,2])               
        name_include.append(sp500[i])
        
#%% Perform PCA        
sp_price_daily=np.array(sp_price_daily).T 
sp_volume_daily=np.array(sp_volume_daily).T 
sp_pm_ratio_daily=np.array(sp_pm_ratio_daily).T
sp_pm_daily=np.sign(sp_pm_ratio_daily)                                              
sp_pm_daily[sp_pm_daily==0]=-1                         
name_include=np.array(name_include)   

demean_daily=pylab.demean(sp_price_daily,axis=0)  

pca = PCA(n_components=demean_daily.shape[1])
pca.fit(demean_daily)                       
                     
#%% Perform and plot PCA 1                  
comatrix=np.cov(np.transpose(demean_daily))                         
w, v = LA.eig(comatrix)

v[:,1]=-v[:,1]
plt.close('all')

fig=pylab.figure(figsize=(10, 5))               
ax = fig.add_subplot(111)
ind = np.argpartition(np.abs(v[:,0]), -12)[-12:] # select first 12 
print(name_include[ind])
xa=np.linspace(1,len(w),len(w))
pylab.bar(xa,v[:,0], 0.8, color='b',alpha=0.8)

for i in range(12):
    qqq=name_include[ind[i]] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    ax.text(ind[i]-5, v[ind[i],0],qqq, fontsize=18)
ax.set_xlabel('Stock', fontsize=20)
ax.set_ylabel(r'Covariability signal [$\partial\$$/$\partial$PC]', fontsize=20)    
plt.tight_layout()
ax.set_xlim([0,w.shape[0]])
plt.xticks([500], [''])
ax.set_ylim([-0.7,0.5])
plt.title('PCA1', fontsize=20)
plt.tight_layout()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig('PCA1.png' ,dpi=500)

#%% Perform and plot PCA 2                     

fig=pylab.figure(figsize=(10, 5))               
ax = fig.add_subplot(111)
ind = np.argpartition(np.abs(v[:,1]), -12)[-12:] # select first 12 
print(name_include[ind])
xa=np.linspace(1,len(w),len(w))
pylab.bar(xa,v[:,1], 0.8, color='b',alpha=0.8)

for i in range(12):
    qqq=name_include[ind[i]] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    ax.text(ind[i]-5, v[ind[i],1],qqq, fontsize=18)
ax.set_xlabel('Stock', fontsize=20)
ax.set_ylabel(r'Covariability signal [$\partial\$$/$\partial$PC]', fontsize=20)    
plt.tight_layout()
ax.set_xlim([0,w.shape[0]])
plt.xticks([500], [''])
ax.set_ylim([-0.2,0.6])
plt.title('PCA2', fontsize=20)
plt.tight_layout()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig('PCA2.png' ,dpi=500)

#%% Perform and plot PCA 3                      

fig=pylab.figure(figsize=(10, 5))               
ax = fig.add_subplot(111)
ind = np.argpartition(np.abs(v[:,2]), -12)[-12:] # select first 12 
print(name_include[ind])
xa=np.linspace(1,len(w),len(w))
pylab.bar(xa,v[:,2], 0.8, color='b',alpha=0.8)

for i in range(12):
    qqq=name_include[ind[i]] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    ax.text(ind[i]-5, v[ind[i],2],qqq, fontsize=18)
ax.set_xlabel('Stock', fontsize=20)
ax.set_ylabel(r'Covariability signal [$\partial\$$/$\partial$PC]', fontsize=20)    
plt.tight_layout()
ax.set_xlim([0,w.shape[0]])
plt.xticks([500], [''])
ax.set_ylim([-0.3,0.5])
plt.title('PCA3', fontsize=20)
plt.tight_layout()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig('PCA3.png' ,dpi=500)

#%% Perform and plot PCA 4                      

fig=pylab.figure(figsize=(10, 5))               
ax = fig.add_subplot(111)
ind = np.argpartition(np.abs(v[:,3]), -12)[-12:] # select first 12 
print(name_include[ind])
xa=np.linspace(1,len(w),len(w))
pylab.bar(xa,v[:,3], 0.8, color='b',alpha=0.8)

for i in range(12):
    qqq=name_include[ind[i]] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    ax.text(ind[i]-5, v[ind[i],3],qqq, fontsize=18)
ax.set_xlabel('Stock', fontsize=20)
ax.set_ylabel(r'Covariability signal [$\partial\$$/$\partial$PC]', fontsize=20)    
plt.tight_layout()
ax.set_xlim([0,w.shape[0]])
plt.xticks([500], [''])
ax.set_ylim([-0.5,0.5])
plt.title('PCA4', fontsize=20)
plt.tight_layout()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig('PCA4.png' ,dpi=500)

#%% Calculation of Explained Variance from the eigenvalues and plot

eig_pairs = [ (np.abs(w[i]),v[:,i]) for i in range(len(w))]
# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)

tot = sum(w)
var_exp = [(i/tot)*100 for i in sorted(w, reverse=True)]
cum_var_exp = np.cumsum(var_exp) 

# Cumulative explained variance# Variances plot
max_cols = len(cum_var_exp)
fig, ax=plt.subplots(figsize=(8, 5))#plt.figure(figsize=(10, 5))
plt.bar(range(max_cols), var_exp, alpha=0.4, align='center', label='individual explained variance', color = 'g')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.step(range(max_cols), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio', fontsize=15)
plt.xlabel('Principal components', fontsize=15)
plt.legend(loc='best', fontsize=13)
plt.xlim((-0.5,5.5))
plt.ylim((0,100))
ax.tick_params(labelsize=13)

#%% projection on PC
plt.close('all')
fig=pylab.figure(figsize=(8, 5))  
ss=w.shape
ax = fig.add_subplot(111)
First_axis = np.dot((v[:,0].reshape(1,ss[0])), np.transpose((demean_daily)))
Second_axis = np.dot((v[:,1].reshape(1,ss[0])), np.transpose((demean_daily)))
Third_axis = np.dot((v[:,2].reshape(1,ss[0])), np.transpose((demean_daily)))
Fourth_axis = np.dot((v[:,3].reshape(1,ss[0])), np.transpose((demean_daily)))
Fifth_axis = np.dot((v[:,4].reshape(1,ss[0])), np.transpose((demean_daily)))
Six_axis = np.dot((v[:,5].reshape(1,ss[0])), np.transpose((demean_daily)))
Seven_axis = np.dot((v[:,6].reshape(1,ss[0])), np.transpose((demean_daily)))
Eight_axis = np.dot((v[:,7].reshape(1,ss[0])), np.transpose((demean_daily)))
Nine_axis = np.dot((v[:,8].reshape(1,ss[0])), np.transpose((demean_daily)))

fig=pylab.figure(figsize=(10, 5))               
ax = fig.add_subplot(111)
plt.plot(First_axis[0,:],label='PCA1')
plt.plot(Second_axis[0,:],label='PCA2')
plt.plot(Third_axis[0,:],label='PCA3')
plt.plot(Fourth_axis[0,:],label='PCA4')
plt.plot(Fifth_axis[0,:],label='PCA5')
plt.plot(Six_axis[0,:],label='PCA6')
plt.plot(Seven_axis[0,:],label='PCA7')
plt.plot(Eight_axis[0,:],label='PCA8')
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(First_axis[0,:]))
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel('PC',fontsize=20)  
plt.title('Projection on PC', fontsize=20)
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.legend(ncol=2)  

#plt.plot(moving_average(Second_axis.T,50),color ='b', label='50days')
#plt.plot(moving_average(Second_axis.T,200),color ='r', label='200days')

#%% time derivative of PCA projection for 5 days average
average=5

fig=pylab.figure(figsize=(20, 12))               
ax = fig.add_subplot(221)

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA1')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA2')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA3')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA4')

ax.set_xlim([5,340])
ax.set_ylim([-30,30])
plt.xticks([32,82,132,182,232,282,332], ['2007','2008','2009','2010','2011','2012','2013'])
ax.tick_params(labelsize=18)
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel(r'[$\partial$PC/$\partial$5days]', fontsize=20)  
plt.tight_layout()

#%% time derivative of PCA projection for 150 days average

average=150
ax = fig.add_subplot(222)

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA1')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA2')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA3')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA4')
plt.xticks([  0.28,   1.95,   3.62,   5.29,   6.96,   8.63,  10.3 ], ['2007','2008','2009','2010','2011','2012','2013'])
ax.tick_params(labelsize=18)
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel(r'[$\partial$PC/$\partial$Season]', fontsize=20)  
plt.tight_layout()
plt.legend(fontsize=15,loc='lower right',ncol=2)

#%% time derivative of PCA projection for 300 days average

average=300
ax = fig.add_subplot(223)

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA1')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA2')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA3')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA4')
plt.xticks([0.2,0.85,1.5,2.15,2.8,3.45,4.1], ['2007','2008','2009','2010','2011','2012','2013'])
ax.tick_params(labelsize=18)
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel(r'[$\partial$PC/$\partial$Year]', fontsize=20)  
plt.tight_layout()

#%% time derivative of PCA projection for 600 days average
average=600           
ax = fig.add_subplot(224)

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA1')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA2')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA3')

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA4')

plt.xticks([ 0.05,  0.21,  0.37,  0.53,  0.69,  0.85,  1.01], ['2007','2008','2009','2010','2011','2012','2013'])
ax.tick_params(labelsize=18)
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel(r'[$\partial$PC/$\partial$2Years]', fontsize=20)  
plt.tight_layout()

#%% plot 
average=5

PCA_pro_grad=np.zeros((336,8))

fig=pylab.figure(figsize=(20, 12))               
ax = fig.add_subplot(221)

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA1')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,0]=Diff[5:341]


days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA2')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,1]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA3')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,2]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA4')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,3]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Fifth_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA5')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,4]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Six_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA6')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,5]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Seven_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA7')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,6]=Diff[5:341]

days = np.arange(choose_num).reshape(1,choose_num)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, Eight_axis[0,:], 18))
first_smooth_=smooth_first(days)
days = np.arange(choose_num/average).reshape(1,choose_num/average)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(first_smooth_[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot((np.diff(first_smooth[0,:])),label='PCA8')

Diff=np.diff(first_smooth[0,:])
PCA_pro_grad[:,7]=Diff[5:341]

ax.set_xlim([5,340])
ax.set_ylim([-30,30])
plt.xticks([32,82,132,182,232,282,332], ['2007','2008','2009','2010','2011','2012','2013'])
ax.tick_params(labelsize=18)
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel(r'[$\partial$PC/$\partial$5days]', fontsize=20)  
plt.tight_layout()
plt.legend()
#plt.xticks([500], [''])
#ax.set_ylim([-0.5,0.5])

#%%   Doing PCA of PCA_pro_grad and Cluster 

comatrix=np.cov(np.transpose(PCA_pro_grad))                         
w_pca, v_pca = LA.eig(comatrix)

eig_pairs = [ (np.abs(w_pca[i]),v_pca[:,i]) for i in range(len(w_pca))]
# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(w_pca)
var_exp = [(i/tot)*100 for i in sorted(w_pca, reverse=True)]
cum_var_exp = np.cumsum(var_exp) 
# Cumulative explained variance# Variances plot

max_cols = len(cum_var_exp)
fig, ax=plt.subplots(figsize=(8, 5))#plt.figure(figsize=(10, 5))
plt.bar(range(max_cols), var_exp, alpha=0.4, align='center', label='individual explained variance', color = 'g')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.step(range(max_cols), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio', fontsize=15)
plt.xlabel('Principal components', fontsize=15)
plt.title('PCA of PCA rate of stock price',fontsize=15)
plt.legend(loc='best', fontsize=13)
plt.xlim((-0.5,5.5))
plt.ylim((0,100))
ax.tick_params(labelsize=13)

#%%
ss=w_pca.shape
First_axis_pca = np.dot((v_pca[:,0].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Second_axis_pca = np.dot((v_pca[:,1].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Third_axis_pca = np.dot((v_pca[:,2].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Fourth_axis_pca = np.dot((v_pca[:,3].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Fifth_axis_pca = np.dot((v_pca[:,4].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Six_axis_pca = np.dot((v_pca[:,5].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Seven_axis_pca = np.dot((v_pca[:,6].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))
Eight_axis_pca = np.dot((v_pca[:,7].reshape(1,ss[0])), np.transpose((PCA_pro_grad)))

start=0
end=336
fig = plt.figure(102)  
ax = fig.add_subplot(111)
ww= end-start
scaled_z=np.arange(ww)/ww + 0.01
colors = plt.cm.coolwarm(scaled_z)
sc=plt.scatter(First_axis_pca[0,start:end],Second_axis_pca[0,start:end], c=colors)
plt.ylabel('PCA1', fontsize=15)
plt.xlabel('PCA2', fontsize=15)
ax.tick_params(labelsize=18)
plt.tight_layout()

start=130
end=336
fig = plt.figure(103)  
ax = fig.add_subplot(111)
ww= end-start
scaled_z=np.arange(ww)/ww + 0.01
colors = plt.cm.coolwarm(scaled_z)
sc=plt.scatter(First_axis_pca[0,start:end],Second_axis_pca[0,start:end], c=colors)
plt.ylabel('PCA1', fontsize=15)
plt.xlabel('PCA2', fontsize=15)
ax.tick_params(labelsize=18)
plt.tight_layout()

#%%

smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(First_axis[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot(np.abs(np.diff(first_smooth[0,:])),label='PCA1')

smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(Second_axis[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot(np.abs(np.diff(first_smooth[0,:])),label='PCA2')

smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(Third_axis[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot(np.abs(np.diff(first_smooth[0,:])),label='PCA3')

smooth_first = np.poly1d(np.polyfit(days[0,:].T, np.mean(Fourth_axis[0,:].reshape(-1, average), axis=1), 20))
first_smooth=smooth_first(days)
plt.plot(np.abs(np.diff(first_smooth[0,:])),label='PCA4')

plt.plot(np.diff(np.mean(First_axis[0,:].reshape(-1, average), axis=1)),label='PCA1')
plt.plot(np.diff(np.mean(Second_axis[0,:].reshape(-1, average), axis=1)),label='PCA2')
plt.plot(np.diff(np.mean(Third_axis[0,:].reshape(-1, average), axis=1)),label='PCA3')
plt.plot(np.diff(np.mean(Fourth_axis[0,:].reshape(-1, average), axis=1)),label='PCA4')
plt.plot(np.diff(np.mean(Fifth_axis[0,:].reshape(-1, average), axis=1)),label='PCA5')
plt.plot(np.diff(np.mean(Six_axis[0,:].reshape(-1, average), axis=1)),label='PCA6')
plt.plot(np.diff(np.mean(Seven_axis[0,:].reshape(-1, average), axis=1)),label='PCA7')
plt.plot(np.diff(np.mean(Eight_axis[0,:].reshape(-1, average), axis=1)),label='PCA8')
plt.legend()  

plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(First_axis[0,:]))
ax.set_xlabel('Time',fontsize=20)
ax.set_ylabel('PC',fontsize=20)  
plt.title('Projection on PC', fontsize=20)
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.legend()  
#plt.plot(moving_average(Second_axis.T,50),color ='b', label='50days')
#plt.plot(moving_average(Second_axis.T,200),color ='r', label='200days')


#%%

days = np.arange(choose_num).reshape(1,1825)
smooth_first = np.poly1d(np.polyfit(days[0,:].T, First_axis[0,:], 18))
first_smooth=smooth_first(days)

smooth_second = np.poly1d(np.polyfit(days[0,:].T, Second_axis[0,:], 18))
second_smooth=smooth_second(days)

smooth_third = np.poly1d(np.polyfit(days[0,:].T, Third_axis[0,:], 18))
third_smooth=smooth_third(days)

smooth_fourth = np.poly1d(np.polyfit(days[0,:].T, Fourth_axis[0,:], 18))
fourth_smooth=smooth_fourth(days)

smooth_fifth = np.poly1d(np.polyfit(days[0,:].T, Fifth_axis[0,:], 18))
fifth_smooth=smooth_fifth(days)

smooth_six = np.poly1d(np.polyfit(days[0,:].T, Six_axis[0,:], 18))
six_smooth=smooth_six(days)

smooth_seven = np.poly1d(np.polyfit(days[0,:].T, Seven_axis[0,:], 18))
seven_smooth=smooth_seven(days)

smooth_eight = np.poly1d(np.polyfit(days[0,:].T, Eight_axis[0,:], 18))
eight_smooth=smooth_eight(days)

smooth_nine = np.poly1d(np.polyfit(days[0,:].T, Nine_axis[0,:], 18))
nine_smooth=smooth_nine(days)

plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))

fig=pylab.figure(figsize=(8, 5))  
ax = fig.add_subplot(111)
#plt.style.use('ggplot')
plt.plot(first_smooth[0,:],label='PCA1')
plt.plot(second_smooth[0,:],label='PCA2')
plt.plot(third_smooth[0,:],label='PCA3')
plt.plot(fourth_smooth[0,:],label='PCA4')
plt.plot(fifth_smooth[0,:],label='PCA5')
plt.plot(six_smooth[0,:],label='PCA6')
plt.plot(seven_smooth[0,:],label='PCA7')
plt.plot(eight_smooth[0,:],label='PCA8')
plt.plot(nine_smooth[0,:],label='PCA9')
#plt.plot(np.diff(Second_axis[0,:]),label='PCA2')
#plt.plot(np.diff(Third_axis[0,:]),label='PCA3')
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('Projection on component',fontsize=15)  
plt.legend()  
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))

fig=pylab.figure(figsize=(8, 5))  
ax = fig.add_subplot(111)
#plt.style.use('ggplot')
plt.plot(np.diff(first_smooth[0,:]),label='PCA1')
plt.plot(np.diff(second_smooth[0,:]),label='PCA2')
plt.plot(np.diff(third_smooth[0,:]),label='PCA3')
plt.plot(np.diff(fourth_smooth[0,:]),label='PCA4')
plt.plot(np.diff(fifth_smooth[0,:]),label='PCA5')
plt.plot(np.diff(six_smooth[0,:]),label='PCA6')
plt.plot(np.diff(seven_smooth[0,:]),label='PCA7')
plt.plot(np.diff(eight_smooth[0,:]),label='PCA8')
plt.plot(np.diff(nine_smooth[0,:]),label='PCA9')
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('Projection on component gradient',fontsize=15)  
plt.legend()  
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))
#plt.ylim(-15,15)

fig=pylab.figure(figsize=(8, 5))  
ax = fig.add_subplot(111)
plt.plot(np.diff(first_smooth[0,:])/np.diff(second_smooth[0,:]),'k*')
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('Rate ratio PCA1&2',fontsize=15)  
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))
plt.ylim(-2.5,2.5)

fig=pylab.figure(figsize=(8, 5))  
ax = fig.add_subplot(111)
plt.plot(np.diff(third_smooth[0,:])/np.diff(second_smooth[0,:]),'k*')
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('Rate ratio PCA1&2',fontsize=15)  
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))
plt.ylim(-2.5,2.5)

fig=pylab.figure(figsize=(8, 5))  
ax = fig.add_subplot(111)
plt.plot(np.diff(fourth_smooth[0,:])/np.diff(second_smooth[0,:]),'k*')
ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('Rate ratio PCA1&2',fontsize=15)  
plt.xticks([162,413,666,918,1170,1422,1672], ['2007','2008','2009','2010','2011','2012','2013'])
plt.xlim(0,len(first_smooth[0,:]))
plt.ylim(-7.5,7.5)
#%%
## plot scatter of PCA1 AND 2
#fig = plt.figure(2)
#ax = fig.add_subplot(222)
#scaled_z=np.arange(choose_num)/choose_num + 0.01
#colors = plt.cm.coolwarm(scaled_z)
#plt.scatter(Third_axis[0,:],Second_axis[0,:], c=colors)
##plt.scatter(Third_axis[0,0:2262],Second_axis[0,0:2262], c=colors)
#
## plot scatter of PCA1 AND 2
#fig = plt.figure(2)
#ax = fig.add_subplot(221)
#scaled_z=np.arange(choose_num)/choose_num + 0.01
#colors = plt.cm.coolwarm(scaled_z)
##plt.scatter(First_axis[0,0:2262],Second_axis[0,0:2262], c=colors)
#plt.scatter(First_axis[0,:],Second_axis[0,:], c=colors)



# plot scatter of apple and C 
#
#fig = plt.figure(2)
#ax = fig.add_subplot(221)
#scaled_z=np.arange(choose_num)/choose_num + 0.01
#colors = plt.cm.coolwarm(scaled_z)
##plt.scatter(First_axis[0,0:2262],Second_axis[0,0:2262], c=colors)
#plt.scatter(sp_price_daily[:,171],sp_price_daily[:,125], c=colors)

#%% Only select first 12 important stocks [tech+finance]
###sselected 
###sselected 
###sselected 

select_daily=demean_daily[:,ind]
pca = PCA(n_components=select_daily.shape[1])
pca.fit(select_daily)
                          
comatrix_select=np.cov(np.transpose(select_daily))                         
w, v = LA.eig(comatrix_select)                   

fig = plt.figure(1)
ax = fig.add_subplot(111)
#ind = np.argpartition(np.abs(v[:,0]), -12)[-12:] # select first 12 
print(name_include[ind])
plt.plot(v[:,0],'r')
plt.plot(v[:,1],'b')

for i in range(12):
    qqq=name_include[ind[i]] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    ax.text(i, v[i,0],qqq, fontsize=12)
ax.set_xlabel('Stock')
ax.set_ylabel('Covariability signal')    
plt.tight_layout()
#plt.savefig('Covariability_signal_1.pdf' ,dpi=500)
First_axis = np.dot((v[:,0].reshape(1,12)), np.transpose((select_daily)))
Second_axis = np.dot((v[:,1].reshape(1,12)), np.transpose((select_daily)))
from scipy import signal
First_axis=scipy.signal.detrend(First_axis)
Second_axis=scipy.signal.detrend(Second_axis)
plt.plot(First_axis[0,:],'b')
plt.plot(-Second_axis[0,:],'r')

fig = plt.figure(3)       
ax = fig.add_subplot(111)
plt.scatter(First_axis[0,:],Second_axis[0,:])
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
###sselected 
###sselected 
###sselected 



#%% 
#####################################################    
#####################################################  
#####       plot_golden_rule_death_rule       #######  
#####################################################  
#####################################################  

def moving_average(array,num):
    ss=array.shape;
    moving_array=np.zeros(ss[0])
    for i in range(num-1,ss[0]):
        moving_array[i]=np.mean(array[i-num+1:i+1]) 
    return moving_array


days=np.array(range(1825))
fig = plt.figure(4)
ax = fig.add_subplot(111)
t_daily_price=sp_price_daily[:,name_include=='table_aapl.csv']
plt.plot(days,t_daily_price,color ='k', label='stock price')
plt.plot(days,moving_average(t_daily_price,50),color ='b', label='50days')
plt.plot(days,moving_average(t_daily_price,200),color ='r', label='200days')

f=moving_average(t_daily_price,50)
g=moving_average(t_daily_price,200)

idx = np.argwhere(np.diff(np.sign(f - g)) != 0).reshape(-1) + 0
                 
#f[idx]-g[idx] > 0 indicates 50<200 death cross 
#f[idx]-g[idx] < 0 indicates 50>200 golden cross  

cross_sign=np.sign(f[idx]-g[idx]) # positive death, negative golden                
death=idx[cross_sign==1]  
golden=idx[cross_sign==-1]                                   
plt.plot(days[death],f[death], 'r*',markersize=20, label='Death-sell')
plt.plot(days[golden],f[golden], 'g*',markersize=20, label='Golden-buy')

if np.max(golden)>np.max(death):
    ss=golden.shape
    real_golden=golden[0:ss[0]-1]
else:
    real_golden=golden[:]

if np.min(golden)>np.min(death):
    ss=death.shape
    real_death=death[1:ss[0]]
else:    
    real_death=death[:]

golden_price=t_daily_price[real_golden]
death_price=t_daily_price[real_death]
invest_ratio=death_price/golden_price
np.prod(invest_ratio)

ax.set_xlabel('Days since 2008/08')
ax.set_ylabel('Stock price of STJ')
ax.set_xlim([250,1750])
ax.set_ylim([np.min(t_daily_price[250:1750,0])-5,np.max(t_daily_price[250:1750,0])+5])
plt.legend()
plt.tight_layout()

#%%

output_cross_profit=[]
##### whole picture 
#golden_rule_death_rule_total_revenue

invest_ratio_summary=np.zeros(467)
average_distance=np.zeros(467)
for i in range(467):
    days=np.array(range(1825))
    t_daily_price=sp_price_daily[:,i]
    f=moving_average(t_daily_price,50)
    g=moving_average(t_daily_price,200)

    idx = np.argwhere(np.diff(np.sign(f - g)) != 0).reshape(-1) + 0
                 
                     #f[idx]-g[idx] > 0 indicates 50<200 death cross 
                       #f[idx]-g[idx] < 0 indicates 50>200 golden cross  

    cross_sign=np.sign(f[idx]-g[idx]) # positive death, negative golden                
    death=idx[cross_sign==1]  
    golden=idx[cross_sign==-1]                                   

    if np.max(golden)>np.max(death):
        ss=golden.shape
        real_golden=golden[0:ss[0]-1]
    else:
         real_golden=golden[:]

    if np.min(golden)>np.min(death):
            ss=death.shape
            real_death=death[1:ss[0]]
    else:    
         real_death=death[:]

    golden_price=t_daily_price[real_golden]
    death_price=t_daily_price[real_death]
    invest_ratio=death_price/golden_price
    invest_ratio_summary[i]=np.prod(invest_ratio)
    average_distance[i]=np.mean(real_death-real_golden)
    
    distance_between_golden_death=real_golden[1:]-real_death[0:-1]

    
    output_cross_profit.append(invest_ratio[1,:])
    
    
    
##   
invest_ratio_summary=(invest_ratio_summary-1)*100

from scipy.stats import rankdata
stocks=np.array(range(467))
absolute_name=[]
for i in range(467):
    qqq=name_include[i] 
    qqq=qqq[6:]
    qqq=qqq.rsplit('.', 1)[0]
    absolute_name.append(qqq)

fig = plt.figure(2)       
ax = fig.add_subplot(111)
sort =np.sort(invest_ratio_summary)
pm = plt.bar(stocks,sort[::-1]/5)
ax.set_ylabel('Value change[%]/year')
ax.set_xlabel('Stocks')
ax.set_title('Golden/death cross rule')
for i in range(273):
    pm[i].set_color('g')
for i in range(273,467):
    pm[i].set_color('r')
#ax.set_ylim([-20,60])
plt.savefig('cross_sp500.pdf' ,dpi=500)


#### give frequency 
fig = plt.figure(3)       
ax = fig.add_subplot(111)
plt.scatter(1825/average_distance/5,(invest_ratio_summary-1)*100/5);
ax.set_xlim([0,10])
ax.set_ylabel('Value change[%]/year')
ax.set_xlabel('Average cross occurrence/year')
plt.savefig('frequency_value_change.pdf' ,dpi=500)


occurrence_freq =1825/average_distance/5
change_year=(invest_ratio_summary-1)*100/5

N=15
occurrence_bin= np.linspace(0, 10, N, endpoint=True)

total_bin=np.zeros(N)
positive_bin=np.zeros(N)
for i in range(N-1):
    total=(change_year[(occurrence_freq>occurrence_bin[i]) & (occurrence_bin[i+1]>occurrence_freq)])
    positive=(change_year[(occurrence_freq>occurrence_bin[i]) & (occurrence_bin[i+1]>occurrence_freq)& (change_year>0)])

    total_bin[i]= total.shape[0]   
    positive_bin[i]=positive.shape[0]
 
fig = plt.figure(4)   
ax = fig.add_subplot(111)
plt.plot(positive_bin/total_bin, 'k')    
ax.set_ylabel('Fraction of profit decision')
ax.set_xlabel('Average cross occurrence/year')
ax.set_xlim([0,10])
plt.savefig('frequency_profit_decision.pdf' ,dpi=500)

#%%
############################################
### stock trend up or down algorithm     ### 
############################################


positive=np.zeros(467)
negative=np.zeros(467)

for i in range(467):    
    positive[i]=np.sum(sp_pm_daily[sp_pm_daily[:,i]==1,i])
    negative[i]=-np.sum(sp_pm_daily[sp_pm_daily[:,i]==-1,i])

cc_daily_price=np.zeros((467,467))

for i in range(467):
    for j in range(467):
        coeccf=np.corrcoef(sp_price_daily[:,i],sp_price_daily[:,j])
        cc_daily_price[i,j]=coeccf[0,1]
        
def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

train_start=100
train_end=400

test_start=401
test_end=800
sign_correct=np.zeros(467)

for i in range(467): 
    train_input_feature=np.zeros((train_end-train_start,train_start*467))
    test_input_feature=np.zeros((test_end-test_start,train_start*467))
        
    train_output_target=np.reshape(sp_pm_ratio_daily[train_start:train_end,i],(train_end-train_start,1))   
    test_output_target=np.reshape(sp_pm_ratio_daily[test_start:test_end,i],(test_end-test_start,1))
    
    for j in range(train_start,train_end):
        train_input_feature[j-train_start,:]= np.reshape(sp_pm_ratio_daily[j-train_start:j,:], (1,train_start*467))

    for j in range(test_start,test_end):
        test_input_feature[j-test_start,:]= np.reshape(sp_pm_ratio_daily[j-train_start:j,:], (1,train_start*467))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.00001,max_iter=1000)
    clf.fit(train_input_feature, train_output_target)
    plt.plot(clf.coef_)
    weighting=np.reshape(clf.coef_,(train_start*467,1))
    
    test_model_output=np.dot(test_input_feature,weighting)
        
    print(np.count_nonzero(np.sign(test_model_output)-np.sign(test_output_target)))
    sign_correct[i]=np.count_nonzero(np.sign(test_model_output)-np.sign(test_output_target))


## we find 20 stocks include itself 50 days before 
## train 1 year and test 3 year 

monthly_days_average_sp_ratio=groupedAvg(sp_pm_daily,10)

train_start=5
train_end=40

test_start=40
test_end=100
sign_correct=np.zeros(467)

for i in range(467): 
    train_input_feature=np.zeros((train_end-train_start,train_start*467))
    test_input_feature=np.zeros((test_end-test_start,train_start*467))
        
    train_output_target=np.reshape(monthly_days_average_sp_ratio[train_start:train_end,i],(train_end-train_start,1))   
    test_output_target=np.reshape(monthly_days_average_sp_ratio[test_start:test_end,i],(test_end-test_start,1))
    
    for j in range(train_start,train_end):
        train_input_feature[j-train_start,:]= np.reshape(monthly_days_average_sp_ratio[j-train_start:j,:], (1,train_start*467))

    for j in range(test_start,test_end):
        test_input_feature[j-test_start,:]= np.reshape(monthly_days_average_sp_ratio[j-train_start:j,:], (1,train_start*467))


    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.00001,max_iter=1000)
    clf.fit(train_input_feature, train_output_target)
    plt.plot(clf.coef_)
    weighting=np.reshape(clf.coef_,(train_start*467,1))
    
    test_model_output=np.dot(test_input_feature,weighting)
        
    print(np.count_nonzero(np.sign(test_model_output)-np.sign(test_output_target)))
    sign_correct[i]=np.count_nonzero(np.sign(test_model_output)-np.sign(test_output_target))


## we find 20 stocks include itself 50 days before 
## train 1 year and test 3 year 

weight=np.array([1.43, 0.51, 0.13, 0.1, 0.07, 0.04, 0.03, 0.023,0.01,0.005,0.0012])
index=np.array([1,2,3,4,5,6,7,8,9,10,11])

fig = plt.figure(2)       
ax = fig.add_subplot(111)
pm = plt.bar(index,weight)
ax.set_ylabel('weighting')
ax.set_xlabel('features')
ax.set_title('Weighting of features determine profitable cross cycle')

plt.savefig('feature weighting.pdf' ,dpi=500)

#%% 3 selected stocks to investigate
                       
from numpy import genfromtxt
first= genfromtxt('table_goog.csv', delimiter=',')
from numpy import genfromtxt
second= genfromtxt('table_aapl.csv', delimiter=',')
from numpy import genfromtxt
third= genfromtxt('table_t.csv', delimiter=',')

time_series=np.zeros((2260,3))
time_series[:,0]=first[:,2]-np.mean(first[:,2] )
time_series[:,1]=second[1666:,2]-np.mean(second[:,2] )
time_series[:,2]=third[1666:,2]-np.mean(third[:,2] )

comatrix=np.cov(np.transpose(time_series))

w, v = LA.eig(comatrix)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(time_series[:,0], time_series[:,1], time_series[:,2], c='k', marker='*')

ax.set_xlabel('Google')
ax.set_ylabel('Apple')
ax.set_zlabel('ATT')

plt.tight_layout()
First_axis = np.dot((v[:,0].reshape(1,3)), np.transpose((time_series)))
Second_axis = np.dot((v[:,1].reshape(1,3)), np.transpose((time_series)))
plt.savefig('Fig1,pdf' ,dpi=500)

fig = plt.figure(2)
plt.scatter(First_axis,Second_axis)
explained_ratio=np.zeros(2)
explained_ratio[0]=w[0]/np.sum(w)
explained_ratio[1]=w[1]/np.sum(w)             
ax = fig.add_subplot(111)
ax.set_xlabel('$PCA1$[88%] \n [0.69, 0.72, 0.02] \n [Google,Apple,ATT] ')
ax.set_ylabel('$PCA2$[11%] \n [0.73, -0.69, 0.01]  \n [Google,Apple,ATT] ')
ax.set_xlim([-600,600])
ax.set_ylim([-600,600])
plt.axhline(0, color='black')
plt.axvline(0, color='black')


first_projection=np.array([0.687*50000,0.726*6400])
second_projection=np.array([0.726*50000,-0.688*6400])
third_projection=np.array([0.023*50000,0.01*6400])


plt.plot(np.array([0,0.687*50000]),np.array([0,0.726*6400]) , 'r-')
plt.plot(np.array([0,0.726*50000]), np.array([0,-0.688*6400]), 'b-')
plt.plot(np.array([0,0.023*50000]), np.array([0,0.01*6400]), 'g-')

plt.tight_layout()
plt.text(601, 70, 'GOOGLE', fontsize=10, color='r')
plt.text(601, -75, 'APPLE', color ='b')
plt.text(601, 25, 'ATT', color ='g')
plt.savefig('Fig2,pdf' ,dpi=500)


fig = plt.figure(3)
plt.scatter(First_axis,Second_axis)
explained_ratio=np.zeros(2)
explained_ratio[0]=w[0]/np.sum(w)
explained_ratio[1]=w[1]/np.sum(w)             
ax = fig.add_subplot(111)
ax.set_xlabel('$PCA1$[88%] \n [0.69, 0.72, 0.02] \n [Google,Apple,ATT] ')
ax.set_ylabel('$PCA2$[11%] \n [0.73, -0.69, 0.01]  \n [Google,Apple,ATT] ')
ax.set_xlim([-600,600])
ax.set_ylim([-200,200])
plt.axhline(0, color='black')
plt.axvline(0, color='black')


first_projection=np.array([0.687*50000,0.726*6400])
second_projection=np.array([0.726*50000,-0.688*6400])
third_projection=np.array([0.023*50000,0.01*6400])


plt.plot(np.array([0,0.687*50000]),np.array([0,0.726*6400]) , 'r-')
plt.plot(np.array([0,0.726*50000]), np.array([0,-0.688*6400]), 'b-')
plt.plot(np.array([0,0.023*50000]), np.array([0,0.01*6400]), 'g-')

plt.tight_layout()
plt.text(601, 70, 'GOOGLE', fontsize=10, color='r')
plt.text(601, -75, 'APPLE', color ='b')
plt.text(601, 25, 'ATT', color ='g')
plt.savefig('Fig3,pdf' ,dpi=500)

fig = plt.figure(4)
ax = fig.add_subplot(111)
plt.plot(time_series[:,0], color='r', label='GOOGLE')
plt.plot(time_series[:,1], color ='b', label='APPLE')
plt.plot(time_series[:,2], color ='g', label='ATT')
ax.set_xlabel('Days since 2004/08')
ax.set_ylabel('Stock price demean')
plt.legend()
plt.tight_layout()
plt.savefig('Fig4,pdf' ,dpi=500)

fig = plt.figure(5)
ax = fig.add_subplot(111)
plt.plot(First_axis.reshape(2260,1), color='k', label='PCA1')
plt.plot(Second_axis.reshape(2260,1), color='b', label='PCA2')
ax.set_xlabel('Days since 2004/08')
ax.set_ylabel('PCA strength')
plt.tight_layout()
plt.savefig('Fig5,pdf' ,dpi=500)

#%% Google-Apple stock 2D analysis  
from numpy import genfromtxt
first= genfromtxt('table_goog.csv', delimiter=',')
from numpy import genfromtxt
second= genfromtxt('table_aapl.csv', delimiter=',')
from numpy import genfromtxt
third= genfromtxt('table_t.csv', delimiter=',')

time_series=np.zeros((2260,3))
time_series[:,0]=first[:,2]-np.mean(first[:,2] )
time_series[:,1]=second[1666:,2]-np.mean(second[:,2] )
time_series[:,2]=third[1666:,2]-np.mean(third[:,2] )

comatrix=np.cov(np.transpose(time_series))
w, v = LA.eig(comatrix)
v=-v
First_axis = np.dot((v[:,0].reshape(1,3)), np.transpose((time_series)))
Second_axis = np.dot((v[:,1].reshape(1,3)), np.transpose((time_series)))

fig = plt.figure(5)
ax = fig.add_subplot(111)
plt.plot(First_axis.reshape(2260,1), color='k', label='PCA1')
plt.plot(Second_axis.reshape(2260,1), color='b', label='PCA2')
ax.set_xlabel('Days since 2004/08')
ax.set_ylabel('PCA strength')
plt.tight_layout()


fig = plt.figure(6)
ax = fig.add_subplot(111)
zzzz1=abs(np.fft.rfft(First_axis))
plt.plot(np.log10(zzzz1[0,1:1000]),'r')
zzzz2=abs(np.fft.rfft(Second_axis))
plt.plot(np.log10(zzzz2[0,1:1000]),'b')

#%%
##### cluster
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:18:58 2017

@author: hhtseng
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from numpy import genfromtxt
import os
from sklearn.decomposition import PCA
import pylab
sp500 = os.listdir("/Users/hhtseng/Documents/Pythonfile/ML/project/daily")
num=len(sp500)
choose_num=1825
sp_price_daily=[]
sp_volume_daily=[]
sp_pm_ratio_daily=[]
name_include=[]

for i in range(500):
    choose= genfromtxt('./daily/' + sp500[i], delimiter=',') # 2:open 3:max 4:min 5:close 6:volume  
    if choose.shape[0]<choose_num:
        print(sp500[i])                   
    else:
        sp_price_daily.append((choose[-choose_num:,3]+choose[-choose_num:,4])/2)
        sp_volume_daily.append(choose[-choose_num:,6])
        sp_pm_ratio_daily.append((choose[-choose_num:,5]-choose[-choose_num:,2])/choose[-choose_num:,2])               
        name_include.append(sp500[i])
        
sp_price_daily=np.array(sp_price_daily).T 
sp_volume_daily=np.array(sp_volume_daily).T 
sp_pm_ratio_daily=np.array(sp_pm_ratio_daily).T
sp_pm_daily=np.sign(sp_pm_ratio_daily)                                              
sp_pm_daily[sp_pm_daily==0]=-1                         
name_include=np.array(name_include)                

demean_daily=pylab.demean(sp_price_daily,axis=0)

#%%  kmeans from sklearn

# get 50 highest variance stock
X=sp_pm_ratio_daily
varX=np.var(X,0)
indexvar50=varX.argsort()[-50:][::-1]
X50=X[:,indexvar50]

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=100, random_state=None).fit(X50)
centerk=kmeans.cluster_centers_

# plot 
nn=X50.shape[1]

fig=pylab.figure()
xa=np.linspace(1,nn,nn)
pylab.bar(xa, centerk[0,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,nn)
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,nn,nn)
pylab.bar(xa, centerk[1,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,nn)
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,nn,nn)
pylab.bar(xa, centerk[2,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,nn)
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,nn,nn)
pylab.bar(xa, centerk[3,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,nn)
plt.tight_layout()
plt.minorticks_on()

fig=pylab.figure()
xa=np.linspace(1,nn,nn)
pylab.bar(xa, centerk[4,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,nn)
#plt.xticks(np.arange(min(xa)-1, max(xa)+1, 10))
plt.tight_layout()
plt.minorticks_on()

#%% k means done by myself
k=5
itera=50

# Fisrt, pick k random points as cluster centers
center_index=np.random.randint(0,len(X),size=k)
center=X[center_index,:]
mindis_5=np.zeros(itera)
iteration=[]

# Then, assign each x to nearest cluster by calculating its distance to each center
for ii in range(0,itera):
    clusters=np.zeros(len(X))
    
    for i in range(0,len(X)):
        dis=[]
        for j in range(0,k):
            dist = np.linalg.norm(X[i]-center[j,:])
            dis.append(dist)
    
        mindis_index=np.argmin(dis)
        clusters[i]=mindis_index # cluster index of each point
        
# Find new cluster center by taking the average of the assigned points
    for i in range(0,k):
        points=[X[j] for j in range(len(X)) if clusters[j] == i]
        center[i]=np.mean(points,axis=0)
        
    for i in range(0,len(X)):
        newd=[]
        for j in range(0,k):
            new_d=np.linalg.norm(X[i]-center[j,:])
            newd.append(new_d)
        
        mindis_5[ii]=mindis_5[ii]+(min(newd))**2
    iteration.append(ii)

fig=pylab.figure()
xa=np.linspace(1,X.shape[1],X.shape[1])
pylab.bar(xa, center[0,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,X.shape[1])
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,X.shape[1],X.shape[1])
pylab.bar(xa, center[1,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,X.shape[1])
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,X.shape[1],X.shape[1])
pylab.bar(xa, center[2,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,X.shape[1])
plt.tight_layout()
plt.minorticks_on()


fig=pylab.figure()
xa=np.linspace(1,X.shape[1],X.shape[1])
pylab.bar(xa, center[3,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,X.shape[1])
plt.tight_layout()
plt.minorticks_on()

fig=pylab.figure()
xa=np.linspace(1,X.shape[1],X.shape[1])
pylab.bar(xa, center[4,:], 0.8, color='blue',alpha=0.4)
plt.xlabel('stock index', fontsize=12)
plt.xlim(1,X.shape[1])
plt.tight_layout()
plt.minorticks_on()


#%% Deep learning 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 00:17:33 2017

@author: hhtseng
"""
###################################
# Deep learning daily stock price #
###################################

import numpy as np
from numpy import linalg as LA
from numpy import genfromtxt
import os
from sklearn.decomposition import PCA
import pylab
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb

sp500 = os.listdir("/Users/hhtseng/Documents/Pythonfile/ML/project/daily/")
num=len(sp500)
choose_num=1825
sp_price_daily=[]
sp_volume_daily=[]
sp_pm_ratio_daily=[]
name_include=[]

for i in range(500):
    choose= genfromtxt('./daily/' + sp500[i], delimiter=',') # 2:open 3:max 4:min 5:close 6:volume  
    if choose.shape[0]<choose_num:
        print(sp500[i])                   
    else:
        sp_price_daily.append((choose[-choose_num:,3]+choose[-choose_num:,4])/2)
        sp_volume_daily.append(choose[-choose_num:,6])
        sp_pm_ratio_daily.append((choose[-choose_num:,5]-choose[-choose_num:,2])/choose[-choose_num:,2])               
        name_include.append(sp500[i])
        
sp_price_daily=np.array(sp_price_daily).T 
sp_volume_daily=np.array(sp_volume_daily).T 
sp_pm_ratio_daily=np.array(sp_pm_ratio_daily).T
sp_pm_daily=np.sign(sp_pm_ratio_daily)                                              
sp_pm_daily[sp_pm_daily==0]=-1                         
name_include=np.array(name_include)
demean_daily=pylab.demean(sp_price_daily,axis=0)

Choose_time=choose[-choose_num:,0]

#%%

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

### select stock ###

stocknum=[33,349,443,125,69,171,108,195,48,397,240,299]

#for iii in range(2,3):
for iii in range(0,len(stocknum)):

    select=stocknum[iii]   ### 125=google , 171=apple, 
    sss=name_include[select]
    ss=sss[6:]
    print(ss)
####################

    # Import price data
    data=sp_price_daily
    
    #data=sp_price_daily
    
    n = data.shape[0]
    p = data.shape[1]
    
    priceindex=data[:,select]
    
    # 80% train 20% test
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end + 1
    test_end = n
    
    data_train_price = data[np.arange(train_start, train_end), :]
    data_test_price = data[np.arange(test_start, test_end), :]
    
    #Choose_time_test = Choose_time[np.arange(test_start, test_end)]
    
    ##############################
    # Import ratio data
    data=sp_pm_ratio_daily
    
    n = data.shape[0]
    p = data.shape[1]
    
    priceindex=data[:,select]
    
    # 80% train 20% test
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end + 1
    test_end = n
    
    data_train_ratio = data[np.arange(train_start, train_end), :]
    data_test_ratio = data[np.arange(test_start, test_end), :]
                
    ###################
    # 3 days
    ###################
    
    #X_train=np.hstack((data_train[0:-3, :], data_train[1:-2,:], data_train[2:-1,:]))
    #y_train = data_train[3:, select]
    #
    #X_test=np.hstack((data_test[0:-3, :], data_test[1:-2,:], data_test[2:-1,:]))
    #y_test = data_test[3:, select]
    
    ###################
    # 2 days
    ###################
    #X_train=np.concatenate((data_train[0:-2, :] , data_train[1:-1,:]), axis=1)
    #y_train = data_train[2:, select]
    
    #X_test=np.concatenate((data_test[0:-2, :] , data_test[1:-1,:]), axis=1)
    #y_test = data_test[2:, select]
    
    ####################
    ## 1 day
    ####################
    #X_train = data_train_price[0:-1, :]
    #y_train = data_train_price[1:, select]
    #
    #X_test = data_test_price[0:-1, :]
    #y_test = data_test_price[1:, select]


    ####################
    ## conbine price & ratio using previous two days(t-1, t-2)
    ####################
    
    X_train=np.hstack((data_train_price[0:-2, :],data_train_price[1:-1, :], data_train_ratio[0:-2, :], data_train_ratio[1:-1, :]   ))
    y_train = data_train_price[2:, select]
    
    X_test=np.hstack((data_test_price[0:-2, :],data_test_price[1:-1, :], data_test_ratio[0:-2, :] , data_test_ratio[1:-1, :] ))
    y_test = data_test_price[2:, select]
    
    #X_train=np.concatenate((data_train_price[0:-1, :] , data_train_ratio[0:-1, :]), axis=1)
    #y_train = data_train_price[1:, select]
    ##
    #X_test=np.concatenate((data_test_price[0:-1, :] , data_test_ratio[0:-1, :]), axis=1)
    #y_test = data_test_price[1:, select]
    
    n_stocks = X_train.shape[1]
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

  
    # revise from deep learning tutorial 
    # https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
    
    ########################
    # Set neurons and layers
    ########################
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()
    
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    
    # Layer 2
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    
    # Layer 3
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    
    # Layer 4
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    
    # Output layer
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    
    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    
    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    
    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    
    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    # Make Session
    net = tf.Session()
    
    # Run initializer
    net.run(tf.global_variables_initializer())
    
    # Number of epochs and batch size
    epochs = 300 ###
    batch_size = 200
    
    trainaccu=[]
    testaccu=[]
    
    for e in range(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
    
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
            
            
            pred_train = net.run(out, feed_dict={X: X_train})
            predict_train=pred_train.T.reshape(pred_train.shape[1],)
            
            
            pred = net.run(out, feed_dict={X: X_test})
            predict=pred.T.reshape(pred.shape[1],)
            
            
            mse=np.sqrt(np.sum((y_test - predict)**2)/len(y_test))
            #print(mse)
            
            if mse<0.01: #### google:mse<18.5, apple:mse<
                print(mse)
                # Prediction            
                fig, ax=plt.subplots(figsize=(8.5, 5))
                
                #line1, = ax1.plot(y_test)
                plt.plot(y_test,label='Real')            
                plt.plot(predict,label='Predict')
                plt.legend(loc='best', fontsize=14)
    
                #line2.set_ydata(aa)
                #line2, = ax1.plot(pred.T)
                #plt.title('Google',fontsize=15)                                    ############
                file_name = ss+'img_epoch_' + str(e) + '_batch_' + str(i) + '.png'
                
                ax.tick_params(labelsize=13.5)
                ax.set_xlabel('Time', fontsize=15)
                ax.set_ylabel('Price', fontsize=15)
                plt.xlim((0,len(y_test)))
                
                plt.xticks([2,65,130,191,251,315], ['2012 Mar','2012 Jun','2012 Sep','2012 Dec',
                           '2013 Mar','2013 June'])
                xymin=min(min(y_test),min(predict))
                xymax=max(max(y_test),max(predict))
                plt.text(295,xymin+2, 'MSE=%s'%round(mse,2), fontsize = 15, color="g")
                plt.savefig(file_name)
                
                #####
                fig, ax =plt.subplots(figsize=(6, 6))
                #xx=np.linspace(1,len(y_test),len(y_test))
                xx = np.linspace(xymin, xymax, 3)
                yy = np.linspace(xymin, xymax, 3)
                ax.plot(xx,yy,"r--")
                plt.scatter(y_test,predict,c='k',alpha=0.5,s=30)
                ax.set_xlabel('Predict price', fontsize=15)
                ax.set_ylabel('The real price', fontsize=15)
                ax.tick_params(labelsize=15)
                plt.tight_layout()
    
                plt.xlim((xymin,xymax))
                plt.ylim((xymin,xymax))
    
                file_name = ss+'scatter_'+'img_epoch_' + str(e) + '_batch_' + str(i) + '.png'
                plt.savefig(file_name)        
                
                #stop
        #print('mse'=mse)
    
        ### positive negative         
        true_pn=np.sign(np.diff(y_train)) 
        #pred_pn=np.sign((predict_train[1:]-y_train[0:-1])) ##
        pred_pn=np.sign(np.diff(predict_train))
        cal0=true_pn+pred_pn
        pn_accuracy=np.count_nonzero(cal0)/len(cal0)
        print('train+-accu=',pn_accuracy)
        trainaccu.append(pn_accuracy)
        
        ### positive negative         
        true_pn=np.sign(np.diff(y_test))
        #pred_pn=np.sign((predict[1:]-y_test[0:-1])) ##
        pred_pn=np.sign(np.diff(predict))
        cal0=true_pn+pred_pn
        pn_accuracy=np.count_nonzero(cal0)/len(cal0)
        print('test+-accu=',pn_accuracy)
        testaccu.append(pn_accuracy)
    
    # write csv
    df = pd.DataFrame({"Train accuracy":trainaccu,"Test accuracy":testaccu,"stock":ss})
    filename=ss+'stock_accuracy'+'.csv'
    #df.to_csv(filename, index=False)
    
#fig, ax=plt.subplots(figsize=(8, 5))
#plt.plot(trainaccu,label='train')
#plt.plot(testaccu,label='test')
#plt.legend(loc='best', fontsize=14)
#ax.set_xlabel('epoch', fontsize=15)
#ax.set_ylabel('Accuracy', fontsize=15)
#ax.tick_params(labelsize=15)
#plt.tight_layout()
#plt.ylim((0.5,1))
#plt.savefig('Google_1day_accu.png')
    
#%%
######################################
# Plot deep learning output .csv 
######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import glob
import os
from os import listdir
import pylab

csvpath='/Users/hhtseng/Documents/Pythonfile/ML/project/'
csv_name = [os.path.basename(x) for x in glob.glob(csvpath+'*_accuracy.csv')]

fig, ax =plt.subplots(figsize=(9, 7))
for i in range(0,len(csv_name)):
    df= pd.read_csv(str(csv_name[i]))
    dff=df.values
    
    test=dff[:,0]
    name=dff[0,2]
    name=name.rsplit('.', 1)[0]
    
    plt.plot(test,label=name,alpha=0.9) 
    plt.legend(loc="upper right", fontsize=18,ncol=3)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Test Accuracy', fontsize=18)
    ax.tick_params(labelsize=16)
    plt.tight_layout()

    plt.xlim((0,300))
    plt.ylim((0.45,0.65))
    plt.tight_layout()
    file_name = 'deep_test_9stock.png'
    #plt.savefig(file_name) 

#%%
M=[]
n=[]
for i in range(0,len(csv_name)):
    df= pd.read_csv(str(csv_name[i]))
    dff=df.values
    
    test=dff[:,0]
    name=dff[0,2]
    name=name.rsplit('.', 1)[0]
    
    maxtest=max(test)
    M.append(maxtest)
    n.append(name)

SortM=sorted(M, reverse=True)
Sortind=sorted(range(len(M)), key=lambda k: M[k], reverse=True)

for i in range(0,len(Sortind)):
    n[i]=n[Sortind[i]]

fig=pylab.figure(figsize=(8, 5))
xa=np.linspace(1,9,9)
pylab.bar(xa, SortM, color='b',alpha=0.3)

plt.xlabel('Stock', fontsize=18)
plt.xticks([1,2,3,4,5,6,7,8,9], [n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8]])
plt.ylabel('Max test accuracy', fontsize=18)
plt.tick_params(labelsize=17)
xx = np.linspace(0, 10, 3)
yy = np.linspace(0.5, 0.5, 3)
plt.plot(xx,yy,"r--")

plt.xlim((0.5,9.5))
plt.ylim((0.475,0.65))

plt.tight_layout()
file_name = 'deep_maxtest_9stock.png'
#plt.savefig(file_name) 

#%%
fig, ax =plt.subplots(figsize=(9, 7))
for i in range(0,len(csv_name)):
    df= pd.read_csv(str(csv_name[i]))
    dff=df.values
    
    train=dff[:,1]
    name=dff[0,2]
    name=name.rsplit('.', 1)[0]
    
    plt.plot(train,label=name,alpha=0.9) 
    plt.legend(loc="lower right", fontsize=18,ncol=3)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Train Accuracy', fontsize=18)
    ax.tick_params(labelsize=16)
    plt.tight_layout()

    plt.xlim((0,300))
    plt.ylim((0.8,1))
    plt.tight_layout()
    file_name = 'deep_train_9stock.png'
    #plt.savefig(file_name)