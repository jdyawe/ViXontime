#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from jqdatasdk import *
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import calendar
import datetime
import time
import datetime as dt




def get_option_info(thetime):
    q=query(opt.OPT_DAILY_PREOPEN.code,opt.OPT_DAILY_PREOPEN.name, opt.OPT_DAILY_PREOPEN.contract_type,opt.OPT_DAILY_PREOPEN.exercise_price, opt.OPT_DAILY_PREOPEN.exercise_date,).filter((opt.OPT_DAILY_PREOPEN.date==thetime[:10]) & (opt.OPT_DAILY_PREOPEN.underlying_symbol == '510050.XSHG'))
    df=opt.run_query(q)
    df = df[~df['name'].str.contains('A')]
    exer = sorted(df['exercise_date'].unique())[:2]
    etf = np.array([get_price('510050.XSHG', count = 1, end_date=thetime, frequency='1m', fields=['close'], skip_paused=False, fq=None)['close'].iloc[0]] * df.shape[0])
    df['dif'] = round(df['exercise_price'] - etf,3)
    df['abs_dif'] = abs(df['exercise_price'] - etf)
    
    atm = df[df['exercise_date'] == exer[0]].sort_values(by = ['abs_dif'])['exercise_price'].iloc[0]
    df = df.drop(columns = ['abs_dif'])
    
    near_call = df[(df['exercise_date'] == exer[0]) & (df['contract_type'] == 'CO') & (df['dif'] >= 0)].sort_values(by = ['exercise_price'], ascending = False)
    near_put = df[(df['exercise_date'] == exer[0]) & (df['contract_type'] == 'PO') & (df['dif'] < 0)].sort_values(by = ['exercise_price'], ascending = False)
    near_ = pd.concat([near_call, near_put])
    near_.loc[near_['exercise_price']==atm,'contract_type'] = 'ATM'
    
    next_call = df[(df['exercise_date'] == exer[1]) & (df['contract_type'] == 'CO') & (df['dif'] >= 0)].sort_values(by = ['exercise_price'], ascending = False)
    next_put = df[(df['exercise_date'] == exer[1]) & (df['contract_type'] == 'PO') & (df['dif'] < 0)].sort_values(by = ['exercise_price'], ascending = False)
    next_ = pd.concat([next_call, next_put])
    next_.loc[next_['exercise_price']==atm,'contract_type'] = 'ATM'
    
    atm_op = df[df['exercise_price'] == atm].sort_values(by = ['exercise_date']).iloc[:4,:]
    
    return near_.set_index('code'), next_.set_index('code'), atm_op.set_index('code')



def get_op_price(thetime, df):
    near_, next_, atm_op = df
    op_price_near = get_price(list(near_.index), count=1, end_date=thetime, frequency='1m')['close'].T
    op_price_next = get_price(list(next_.index), count=1, end_date=thetime, frequency='1m')['close'].T
    op_price_near.columns, op_price_next.columns  = ['current_price'], ['current_price']
    near_p, next_p = pd.concat([near_,op_price_near],axis = 1), pd.concat([next_,op_price_next],axis = 1)
    near_p, next_p = near_p[near_p['current_price'] > 0], next_p[next_p['current_price'] > 0]
    
    atm_price_near = get_price(list(atm_op.index[:2]), count=1, end_date=thetime, frequency='1m')['close'].T.iloc[:,0].mean()
    atm_price_next = get_price(list(atm_op.index[2:]), count=1, end_date=thetime, frequency='1m')['close'].T.iloc[:,0].mean()

    near_p.loc[near_p['contract_type']=='ATM','current_price'], next_p.loc[next_p['contract_type']=='ATM','current_price'] = atm_price_near,atm_price_next
  
    return near_p, next_p


def T(thetime, df):
    def convert_time(thetime):
        return [int(i) for i in thetime.split(' ')[0].split('-') + thetime.split(' ')[1].split(':')]

    time1, time2, time3 = convert_time(thetime), convert_time(str(df[0]['exercise_date'].iloc[0]) + ' 15:00:00'), convert_time(str(df[1]['exercise_date'].iloc[0]) + ' 15:00:00') 

    t1 = dt.datetime(time1[0],time1[1],time1[2],time1[3],time1[4],time1[5])
    t2 = dt.datetime(time2[0],time2[1],time2[2],time2[3],time2[4],time2[5])
    t3 = dt.datetime(time3[0],time3[1],time3[2],time3[3],time3[4],time3[5])
    t_mid = dt.datetime(time1[0],time1[1],time1[2],23,59,59)

    m_currentday = int((t_mid - t1).total_seconds() / 60)
    m_settlement = 570
    m_otherday_near = int((t2 - t1).total_seconds() / 60)
    m_otherday_next = int((t3 - t1).total_seconds() / 60)

    return (m_currentday + m_settlement + m_otherday_near) / 525600, (m_currentday + m_settlement + m_otherday_next) / 525600


def TN(thetime, df):
    def convert_time(thetime):
        return [int(i) for i in thetime.split(' ')[0].split('-') + thetime.split(' ')[1].split(':')]

    time1, time2, time3 = convert_time(thetime), convert_time(str(df[0]['exercise_date'].iloc[0]) + ' 15:00:00'), convert_time(str(df[1]['exercise_date'].iloc[0]) + ' 15:00:00') 

    t1 = dt.datetime(time1[0],time1[1],time1[2],time1[3],time1[4],time1[5])
    t2 = dt.datetime(time2[0],time2[1],time2[2],time2[3],time2[4],time2[5])
    t3 = dt.datetime(time3[0],time3[1],time3[2],time3[3],time3[4],time3[5])

    nt1 = int((t2 - t1).total_seconds() / 60)
    nt2 = int((t3 - t1).total_seconds() / 60)
    
    return nt1, nt2


def get_forward(thetime, df):
    df = df[2]
    f_near = get_price(df.index[0], count=1, end_date=thetime, frequency='1m')['close'].iloc[0] - get_price(df.index[1], count=1, end_date=thetime, frequency='1m')['close'].iloc[0] + df['exercise_price'].iloc[0]
    f_next = get_price(df.index[2], count=1, end_date=thetime, frequency='1m')['close'].iloc[0] - get_price(df.index[3], count=1, end_date=thetime, frequency='1m')['close'].iloc[0] + df['exercise_price'].iloc[0]
    return f_near, f_next


def get_vix(thetime, df):
    df1, df2 = get_op_price(thetime, df)
    f1, f2 = get_forward(thetime, df)
    t1, t2 = T(thetime, df)
    k0 = df1[df1['contract_type'] == 'ATM']['exercise_price'].iloc[0]
    nt1, nt2 = TN(thetime, df)
    n30, n365 = 43200, 525600

    dif_near, dif_next = [],[]
    
    totalsec = df1['exercise_price']
    dif_near = [round((totalsec.iloc[i-1]-totalsec.iloc[i+1])/2,3) for i in range(1,len(df1['exercise_price'][1:-1])+1)]
    dif_near.insert(0,round(totalsec.iloc[0]-totalsec.iloc[1],3))
    dif_near.insert(-1,round(totalsec.iloc[-2]-totalsec.iloc[-1],3))
    
    totalsec = df2['exercise_price']
    dif_next = [round((totalsec.iloc[i-1]-totalsec.iloc[i+1])/2,3) for i in range(1,len(df2['exercise_price'][1:-1])+1)]
    dif_next.insert(0,round(totalsec.iloc[0]-totalsec.iloc[1],3))
    dif_next.insert(-1,round(totalsec.iloc[-2]-totalsec.iloc[-1],3))
        
    delta_k1, delta_k2 = np.array(dif_near),np.array(dif_next) 
    k_sqr1, k_sqr2 = np.array(df1['exercise_price'] * df1['exercise_price']), np.array(df2['exercise_price'] * df2['exercise_price'])
    e_term1, e_term2 = [2.718 ** (0.015*t1)] * df1.shape[0], [2.718 ** (0.015*t1)] * df2.shape[0]
    q_k1, q_k2 = np.array(df1['current_price']),np.array(df2['current_price'])
    contribution1, contribution2 = (delta_k1 / k_sqr1) * e_term1 * q_k1, (delta_k2 / k_sqr2) * e_term2 * q_k2
    con1, con2 = contribution1.sum(), contribution2.sum()
    sec_term1, sec_term2 = ((f1/k0-1)**2)*(1/t1), ((f2/k0-1)**2)*(1/t2)
    sigma1, sigma2 = (2/t1) * con1 - sec_term1, (2/t2) * con2 - sec_term2

    vix = ((t1*sigma1*((nt2-n30)/(nt2-nt1)) + t2*sigma2*((n30-nt1)/(nt2-nt1)))*(n365/n30))**0.5 * 100
    return vix





