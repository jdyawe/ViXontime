import jqdatasdk as jds
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import sklearn as skl

# eft复权后取平价合约和直接利用沽购合约价差最小取平价合约的结果不一定相同！！！

pd.set_option('display.max_columns', 30)

start = pd.to_datetime('20190301')
end = pd.to_datetime('20210808')
tdy = datetime.now()
symbol = "510050.XSHG"


class VIXDaily():
    def __init__(self, underlying_asset: str, startdate, enddate):
        # symbol is the underlying assets of option
        # startdate: VIX start time
        # enddate: VIX end time
        self.ID = '18810592263'
        self.pwd = '592263'
        jds.auth(self.ID, self.pwd)
        self.R = 0.02  # 无风险利率 约为1年期中国国债

        self.fields = ['code', 'name', 'exercise_price', 'contract_type', 'list_date']
        self.chosen = ['pre_settle', 'settle_price', 'change_pct_settle', 'pre_close', 'open', 'close', 'volume', ]

        # 筛选历史合约信息
        q = jds.query(jds.opt.OPT_CONTRACT_INFO)\
            .filter((jds.opt.OPT_CONTRACT_INFO.underlying_symbol == underlying_asset))
        self.history_option = jds.opt.run_query(q)

        # history_option.sort_values(by=['list_date', 'trading_code'])
        column = ['code', 'name', 'exercise_price', 'list_date', 'expire_date', 'contract_type']
        self.expire_dates = self.history_option.expire_date.drop_duplicates().sort_values().values
        self.list_dates = self.history_option.list_date.drop_duplicates().sort_values().values
        self.history_option = self.history_option.reindex(columns=column).set_index(['expire_date',])
        # print(history_option)

        #取历史合约行情
        self.tradedays = jds.get_trade_days(start_date=startdate, end_date=enddate)
        # for i in np.arange(len(self.tradedays)):
        #     self.tradedays[i] = datetime(self.tradedays[i].year, self.tradedays[i].month, self.tradedays[i].day, )
        # print(self.tradedays)
        self.Vix = [0]*len(self.tradedays)

        self.getETF(underlying_asset, startdate, enddate, None)

    def start(self):
        # 初始化
        date = self.tradedays[0]
        self.getOption(date)
        self.Vix[0] = self.evaluationByETFClosest(date)

    def forward(self):
        for i in np.arange(1, self.tradedays.size):
            date = self.tradedays[i]
            if date in self.expire_dates or date in self.list_dates:
                self.getOption(date)
            self.Vix[i] = self.evaluationByETFClosest(date)

    def getOption(self, trading_date):
        # 初始化
        self.neartermExpireDate, self.nexttermExpireDate = self.expire_dates[self.expire_dates >= trading_date][:2]

        self.neartermOption = self.history_option.loc[self.neartermExpireDate, :].reindex(
            columns=self.fields).sort_values(by='name')
        self.neartermOption = self.neartermOption[self.neartermOption.list_date <= trading_date]
        self.nexttermOption = self.history_option.loc[self.nexttermExpireDate, :].reindex(
            columns=self.fields).sort_values(by='name')
        self.nexttermOption = self.nexttermOption[self.nexttermOption.list_date <= trading_date]
        self.neartermOptionPrice = pd.DataFrame()
        self.nexttermOptionPrice = pd.DataFrame()
        for Option in self.neartermOption.loc[:, 'code']:
            que = jds.query(jds.opt.OPT_DAILY_PRICE).filter(
                (jds.opt.OPT_DAILY_PRICE.code == Option) & (jds.opt.OPT_DAILY_PRICE.date >= trading_date))
            self.neartermOptionPrice = pd.concat([self.neartermOptionPrice, jds.opt.run_query(que)])
        for Option in self.nexttermOption.loc[:, 'code']:
            que = jds.query(jds.opt.OPT_DAILY_PRICE).filter(
                (jds.opt.OPT_DAILY_PRICE.code == Option) & (jds.opt.OPT_DAILY_PRICE.date >= trading_date))
            self.nexttermOptionPrice = pd.concat([self.nexttermOptionPrice, jds.opt.run_query(que)])
        # for i in self.neartermOptionPrice.index:
        #     for t in np.arange(len(self.neartermOptionPrice.loc[i, 'date'])):
        #         self.neartermOptionPrice.loc[i, 'date'][t] = self.neartermOptionPrice.loc[i, 'date'][t].date()
        # for i in self.nexttermOptionPrice.index:
        #     for t in np.arange(len(self.nexttermOptionPrice.loc[i, 'date'])):
        #         self.nexttermOptionPrice.loc[i, 'date'][t] = self.nexttermOptionPrice.loc[i, 'date'][t].date()

        self.neartermOptionPrice = self.neartermOptionPrice.set_index(['code', 'date']).loc[:, self.chosen]
        self.nexttermOptionPrice = self.nexttermOptionPrice.set_index(['code', 'date']).loc[:, self.chosen]

        # print(neartermOption)
        # print(nexttermOptionPrice.loc['10000841.XSHG', :])

    def evaluationByCPClosest(self, trading_date):
        # 取平价合约
        tempP = self.neartermOption.reset_index().set_index('contract_type').loc[
            'PO', ['exercise_price', 'code', 'name']].set_index('exercise_price', drop=False)
        tempC = self.neartermOption.reset_index().set_index('contract_type').loc[
            'CO', ['exercise_price', 'code', 'name']].set_index('exercise_price')
        nearOptionEval = pd.concat({'C': tempC, 'P': tempP}, axis=1)
        difference = [0] * len(nearOptionEval)
        Q = [0] * len(nearOptionEval)
        DeltaK = [0] * len(nearOptionEval)
        for i in np.arange(len(nearOptionEval)):
            row = nearOptionEval.iloc[i, :]
            difference[i] = self.neartermOptionPrice.loc[((row[('C', 'code')]), pd.to_datetime(trading_date)), 'settle_price'] - \
                            self.neartermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'settle_price']

        nearOptionEval['diff'] = difference
        Snear = nearOptionEval.iloc[np.argmin(np.abs(difference)), :]
        Tnear = (self.neartermExpireDate - trading_date).days / 365
        Fnear = float(Snear.name + np.exp(self.R * Tnear) * Snear['diff'])

        # print(nearOptionEval)

        K0near = nearOptionEval.loc[:, ('P', 'exercise_price')] - Fnear
        if (K0near < 0).any():
            K0near = K0near[K0near < 0]
            K0near = nearOptionEval.loc[K0near.idxmax(), :]
        else:
            K0near = nearOptionEval.loc[K0near.idxmin(), :]
        # print(' - -'*10)
        # print(trading_date)
        # print('当月：')
        # print(K0near)
        # print(Snear)
        # print(Fnear)

        for i in np.arange(len(nearOptionEval)):
            p = K0near.name
            row = nearOptionEval.iloc[i, :]
            # security = row[('C', 'code')] if row.name > p else row[('P', 'code')]
            # bp = jds.get_ticks(security, pd.to_datetime(trading_date) + dt.timedelta(hours=14, minutes=59),
            #                    pd.to_datetime(trading_date) + dt.timedelta(hours=15), fields=['a1_p', 'b1_p'])
            # Q[i] = (bp.iloc[-1, 0] + bp.iloc[-1, 1])/2
            # print(Q[i])
            Q[i] = self.neartermOptionPrice.loc[
                ((row[('C', 'code')]), pd.to_datetime(trading_date)), 'close'] if row.name > p \
                else self.neartermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'close']
            # print(Q[i])

            if i == 0:
                DeltaK[i] = (nearOptionEval.index[i + 1] - nearOptionEval.index[i])
            elif i == len(nearOptionEval) - 1:
                DeltaK[i] = (nearOptionEval.index[i] - nearOptionEval.index[i - 1])
            else:
                DeltaK[i] = (nearOptionEval.index[i + 1] - nearOptionEval.index[i - 1]) / 2
        nearOptionEval['Q'] = Q
        nearOptionEval['DeltaK'] = DeltaK

        tempP = self.nexttermOption.reset_index().set_index('contract_type').loc[
            'PO', ['exercise_price', 'code', 'name']].set_index('exercise_price', drop=False)
        tempC = self.nexttermOption.reset_index().set_index('contract_type').loc[
            'CO', ['exercise_price', 'code', 'name']].set_index('exercise_price')
        nextOptionEval = pd.concat({'C': tempC, 'P': tempP}, axis=1)

        difference = [0] * len(nextOptionEval)
        Q = [0] * len(nextOptionEval)
        DeltaK = [0] * len(nextOptionEval)
        for i in np.arange(len(nextOptionEval)):
            row = nextOptionEval.iloc[i, :]
            difference[i] = self.nexttermOptionPrice.loc[((row[('C', 'code')]), pd.to_datetime(trading_date)), 'settle_price'] - \
                            self.nexttermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'settle_price']

        nextOptionEval['diff'] = difference  # diff 字段用于求F
        Snext = nextOptionEval.iloc[np.argmin(np.abs(difference)), :]
        Tnext = (self.nexttermExpireDate - trading_date).days / 365
        Fnext = float(Snext.name + np.exp(self.R * Tnext) * Snext['diff'])

        K0next = nextOptionEval.loc[:, ('P', 'exercise_price')] - Fnear
        if (K0next < 0).any():
            K0next = K0next[K0next < 0]
            K0next = nextOptionEval.loc[K0next.idxmax(), :]
        else:
            K0next = nextOptionEval.loc[K0next.idxmin(), :]
        # print('下月：')
        # print(K0next)
        # print(Snext)
        # print(Fnext)
        # print(self.etfPrice.loc[trading_date.strftime('%Y-%m-%d'), 'close'])
        # print(' - -' * 10)

        for i in np.arange(len(nextOptionEval)):
            p = K0next.name
            row = nextOptionEval.iloc[i, :]
            # security = row[('C', 'code')] if row.name > p else row[('P', 'code')]
            # bp = jds.get_ticks(security, pd.to_datetime(trading_date) + dt.timedelta(hours=14, minutes=59),
            #                    pd.to_datetime(trading_date) + dt.timedelta(hours=15), fields=['a1_p', 'b1_p'])
            # Q[i] = (bp.iloc[-1, 0] + bp.iloc[-1, 1])/2
            # print(Q[i])
            Q[i] = self.nexttermOptionPrice.loc[
                ((row[('C', 'code')]), pd.to_datetime(trading_date)), 'close'] if row.name > p \
                else self.nexttermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'close']
            # print(Q[i])
            if i == 0:
                DeltaK[i] = (nextOptionEval.index[i + 1] - nextOptionEval.index[i]) / 2
            elif i == len(nextOptionEval) - 1:
                DeltaK[i] = (nextOptionEval.index[i] - nextOptionEval.index[i - 1]) / 2
            else:
                DeltaK[i] = (nextOptionEval.index[i + 1] - nextOptionEval.index[i - 1]) / 2
        nextOptionEval['Q'] = Q
        nextOptionEval['DeltaK'] = DeltaK

        nearOptionEval = nearOptionEval.reset_index()
        nextOptionEval = nextOptionEval.reset_index()

        if Tnear != 0:
            sigma_near = 2 / Tnear * (nearOptionEval.DeltaK / (nearOptionEval.exercise_price ** 2) * np.exp(
                self.R * Tnear) * nearOptionEval.Q).sum() - 1 / Tnear * (Fnear / K0near.name - 1) ** 2
        else:
            sigma_near = 0
        sigma_next = 2 / Tnext * (nextOptionEval.DeltaK / (nextOptionEval.exercise_price ** 2) * np.exp(
            self.R * Tnear) * nextOptionEval.Q).sum() - 1 / Tnext * (Fnext / K0next.name - 1) ** 2

        sigma = 100*np.sqrt(1/30*(Tnear*sigma_near*(Tnext-30)/(Tnext-Tnear)+Tnext*sigma_next*(30-Tnear)/(Tnext-Tnear)))

        return [sigma_near, sigma_next, sigma, nearOptionEval, nextOptionEval]

    def evaluationByETFClosest(self, trading_date):
        # 取平价合约
        tempP = self.neartermOption.reset_index().set_index('contract_type').loc[
            'PO', ['exercise_price', 'code', 'name']].set_index('exercise_price', drop=False)
        tempC = self.neartermOption.reset_index().set_index('contract_type').loc[
            'CO', ['exercise_price', 'code', 'name']].set_index('exercise_price')
        nearOptionEval = pd.concat({'C': tempC, 'P': tempP}, axis=1)
        difference = [0] * len(nearOptionEval)
        Q = [0] * len(nearOptionEval)
        DeltaK = [0] * len(nearOptionEval)
        for i in np.arange(len(nearOptionEval)):
            row = nearOptionEval.iloc[i, :]
            difference[i] = self.neartermOptionPrice.loc[((row[('C', 'code')]), pd.to_datetime(trading_date)), 'settle_price'] - \
                            self.neartermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'settle_price']

        nearOptionEval['diff'] = difference

        etfprice = self.etfPrice.loc[trading_date.strftime('%Y-%m-%d'), 'close']
        etfOptionDiff = nearOptionEval.reset_index().exercise_price - etfprice
        Snear = nearOptionEval.iloc[np.argmin(np.abs(etfOptionDiff)), :]
        Tnear = (self.neartermExpireDate - trading_date).days / 365
        Fnear = float(Snear.name + np.exp(self.R * Tnear) * Snear['diff'])

        del etfprice, etfOptionDiff

        # print(nearOptionEval)

        K0near = nearOptionEval.loc[:, ('P', 'exercise_price')] - Fnear
        if (K0near < 0).any():
            K0near = K0near[K0near < 0]
            K0near = nearOptionEval.loc[K0near.idxmax(), :]
        else:
            K0near = nearOptionEval.loc[K0near.idxmin(), :]
        # print(' - -'*10)
        # print(trading_date)
        # print('当月：')
        # print(K0near)

        for i in np.arange(len(nearOptionEval)):
            p = K0near.name
            row = nearOptionEval.iloc[i, :]
            # security = row[('C', 'code')] if row.name > p else row[('P', 'code')]
            # bp = jds.get_ticks(security, pd.to_datetime(trading_date) + dt.timedelta(hours=14, minutes=59),
            #                    pd.to_datetime(trading_date) + dt.timedelta(hours=15), fields=['a1_p', 'b1_p'])
            # Q[i] = (bp.iloc[-1, 0] + bp.iloc[-1, 1])/2
            # print(Q[i])
            # print(self.neartermOptionPrice)
            Q[i] = self.neartermOptionPrice.loc[
                ((row[('C', 'code')]), pd.to_datetime(trading_date)), 'close'] if row.name > p \
                else self.neartermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'close']
            # print(Q[i])
            if i == 0:
                DeltaK[i] = (nearOptionEval.index[i + 1] - nearOptionEval.index[i])
            elif i == len(nearOptionEval) - 1:
                DeltaK[i] = (nearOptionEval.index[i] - nearOptionEval.index[i - 1])
            else:
                DeltaK[i] = (nearOptionEval.index[i + 1] - nearOptionEval.index[i - 1]) / 2
        nearOptionEval['Q'] = Q
        nearOptionEval['DeltaK'] = DeltaK

        tempP = self.nexttermOption.reset_index().set_index('contract_type').loc[
            'PO', ['exercise_price', 'code', 'name']].set_index('exercise_price', drop=False)
        tempC = self.nexttermOption.reset_index().set_index('contract_type').loc[
            'CO', ['exercise_price', 'code', 'name']].set_index('exercise_price')
        nextOptionEval = pd.concat({'C': tempC, 'P': tempP}, axis=1)

        difference = [0] * len(nextOptionEval)
        Q = [0] * len(nextOptionEval)
        DeltaK = [0] * len(nextOptionEval)
        for i in np.arange(len(nextOptionEval)):
            row = nextOptionEval.iloc[i, :]
            difference[i] = self.nexttermOptionPrice.loc[((row[('C', 'code')]), pd.to_datetime(trading_date)), 'settle_price'] - \
                            self.nexttermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'settle_price']

        nextOptionEval['diff'] = difference  # diff 字段用于求F
        etfprice = self.etfPrice.loc[trading_date.strftime('%Y-%m-%d'), 'close']
        etfOptionDiff = nextOptionEval.reset_index().exercise_price - etfprice
        Snext = nextOptionEval.iloc[np.argmin(np.abs(etfOptionDiff)), :]
        Tnext = (self.nexttermExpireDate - trading_date).days / 365
        Fnext = float(Snext.name + np.exp(self.R * Tnext) * Snext['diff'])

        del etfprice, etfOptionDiff

        K0next = nextOptionEval.loc[:, ('P', 'exercise_price')] - Fnear
        if (K0next < 0).any():
            K0next = K0next[K0next < 0]
            K0next = nextOptionEval.loc[K0next.idxmax(), :]
        else:
            K0next = nextOptionEval.loc[K0next.idxmin(), :]
        # print('下月：')
        # print(K0next)
        # print(self.etfPrice.loc[trading_date.strftime('%Y-%m-%d'), 'close'])
        # print(' - -' * 10)

        for i in np.arange(len(nextOptionEval)):
            p = K0next.name
            row = nextOptionEval.iloc[i, :]
            # security = row[('C', 'code')] if row.name>p else row[('P', 'code')]
            # bp = jds.get_price(security, pd.to_datetime(trading_date)+dt.timedelta(hours=14, minutes=59),
            #                    pd.to_datetime(trading_date)+dt.timedelta(hours=15), fields = ['a1_p', 'b1_p'])
            # Q[i] = (bp.iloc[-1, 0] + bp.iloc[-1, 1])/2
            # print(Q[i])
            Q[i] = self.nexttermOptionPrice.loc[
                ((row[('C', 'code')]), pd.to_datetime(trading_date)), 'close'] if row.name > p \
                else self.nexttermOptionPrice.loc[((row[('P', 'code')]), pd.to_datetime(trading_date)), 'close']
            # print(Q[i])
            if i == 0:
                DeltaK[i] = (nextOptionEval.index[i + 1] - nextOptionEval.index[i]) / 2
            elif i == len(nextOptionEval) - 1:
                DeltaK[i] = (nextOptionEval.index[i] - nextOptionEval.index[i - 1]) / 2
            else:
                DeltaK[i] = (nextOptionEval.index[i + 1] - nextOptionEval.index[i - 1]) / 2
        nextOptionEval['Q'] = Q
        nextOptionEval['DeltaK'] = DeltaK

        nearOptionEval = nearOptionEval.reset_index()
        nextOptionEval = nextOptionEval.reset_index()

        if Tnear != 0:
            sigma_near = 2 / Tnear * (nearOptionEval.DeltaK / (nearOptionEval.exercise_price ** 2) * np.exp(
                self.R * Tnear) * nearOptionEval.Q).sum() - 1 / Tnear * (Fnear / K0near.name - 1) ** 2
        else:
            sigma_near = 0
        sigma_next = 2 / Tnext * (nextOptionEval.DeltaK / (nextOptionEval.exercise_price ** 2) * np.exp(
            self.R * Tnear) * nextOptionEval.Q).sum() - 1 / Tnext * (Fnext / K0next.name - 1) ** 2

        sigma = 100*np.sqrt(1/30*(Tnear*sigma_near*(Tnext-30)/(Tnext-Tnear)+Tnext*sigma_next*(30-Tnear)/(Tnext-Tnear)))

        return [sigma_near, sigma_next, sigma, nearOptionEval, nextOptionEval]

    def getETF(self, security:str, start, end, fqf):
        self.etfPrice = jds.get_price(security, start, end, 'daily', fq=None, panel=False)
        self.etfPrice['change'] = self.etfPrice.close.diff()
        self.etfPrice['pctchange'] = self.etfPrice.change/self.etfPrice.close.shift(1)
        # print(self.etfPrice)

def main():
    K = VIXDaily(underlying_asset=symbol, startdate=start, enddate=end)
    K.start()
    K.forward()
    vix = pd.Series(np.zeros(len(K.Vix)), index=K.tradedays)
    for i in np.arange(len(K.tradedays)):
        day = K.tradedays[i]
        vix[day] = K.Vix[i][2]

    # print(temp)

    # print(sigma_near)
    # print(K0near)
    # print(sigma_near)
    # print(sigma_next)

    plt.plot(vix)
    vix.to_csv('VIXDaily.csv', header=None)
    plt.show()


if __name__ == '__main__':
    main()



