from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import redis
import yaml
import threading
import time
import sys
from collections import defaultdict

from ut_redis import FrRedis
from ut_dmdi import DailyMdi

import multiprocessing

import seaborn as sns

import scipy
import sklearn as sl

from sublist import rKeysList

# 合成期货价格F也许也有不错的意义？？？？？？？？？？？？

# 程序中存在数据标度不同的问题待解决

# 计算中的浮点数精度待处理！！！！！

pd.set_option('display.max_columns', 30)
pd.set_option('mode.chained_assignment', None)


class HistoryThread(threading.Thread):
    def __init__(self, f, startcount, endcount, nearOption, nextOption):
        super(HistoryThread, self).__init__()
        self.tar = f
        self.s = startcount
        self.e = endcount
        self.nearOption = nearOption.copy()
        self.nextOption = nextOption.copy()

    def run(self):
        self.tar(self.s, self.e, self.nearOption, self.nextOption)


class MyThread(threading.Thread):
    '''
    every threading is for a subscriber
    one channel one threading
    every field is saved in the corresponding field in Data:list of dict
    '''

    def __init__(self, threadName: str, counts, channel: str, host, port, passwd, db, ):
        super(MyThread, self).__init__()
        self.threadName = threadName
        self.flag = True
        self.counts = counts  # 指向待更新的行
        self.countsTag = counts  # 指向最后更新的一行
        self.channel = channel
        self.host = host
        self.port = port
        self.db = db
        self.passwd = passwd
        self.key = None
        print(f'{self.threadName} is ready to start!')

    def run(self):
        if self.counts >= 14402:
            print('Bye!')
            return

        conn = redis.Redis(host=self.host, port=self.port, password=self.passwd, charset='gb18030',
                           errors='replace', decode_responses=True, db=self.db)
        s = conn.pubsub()
        s.subscribe(self.channel)
        for item in s.listen():
            if self.flag:
                if item['type'] == 'message':
                    self.key = item['data']
                    # values = conn.hmget(item['data'], self.objects)
                    # g_LineDatas.iloc[self.counts, :] = values
                    self.counts = self.counts + 1  # max --> 14402
                    # if self.counts >= 14402:
                    #     break
            else:
                break

        s.close()
        conn.close()
        if self.counts >= 14402:
            print(f'{self.threadName} finishes its tasks!')
        else:
            print(f'{self.threadName} is killed by murder!')

    def followup(self):
        self.countsTag = self.counts

    def stop(self):
        self.flag = False


tdy = datetime.now().replace(hour=9, minute=30, second=0)

symbol = "510050.XSHG"


class VIXontime():
    def __init__(self, symbol, History=True):
        self._Sr_vixOntime = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixOntimeC = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixOntimeP = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNearOntime = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNearOntimeC = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNearOntimeP = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNextOntime = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNextOntimeC = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNextOntimeP = pd.Series([np.nan] * 14402, dtype='float64')

        self.symbol = symbol
        self.timestampinit()
        self.serverConn()
        self.ThreadMonitor = MyThread(threadName='JJ', counts=self.counts, channel='H:',
                                      host='168.36.1.181', port=6379, passwd='', db=9)
        self.ThreadMonitor.start()

        self.R = 0.015  # 无风险利率 约为1年期中国国债
        # row: timestamp columns: 计算波动率所需的中间量

        self.OptionGetFilter()
        self.OPS = self.nearOption.src.values.tolist() + self.nextOption.src.values.tolist()
        self.OPSidx = self.nearOption.index.tolist() + self.nextOption.index.tolist()

        TS1 = pd.date_range(start=tdy, end=tdy + dt.timedelta(hours=2), freq='1s')
        TS2 = pd.date_range(start=tdy.replace(hour=13, minute=0, second=0), end=tdy.replace(hour=15, minute=00),
                            freq='1s')
        TS = TS1.append(TS2)
        self.TminRemainingNear = [round(t.total_seconds() // 60 / (365 * 1440), 4) for t in
                                  TS.__rsub__(self.nearexpire)]
        # print(self.TminRemainingNear)
        self.TminRemainingNext = [round(t.total_seconds() // 60 / (365 * 1440), 4) for t in
                                  TS.__rsub__(self.nextexpire)]
        del TS1, TS2, TS

        # self.HistoryData()
        # self.evaluation()

        time.sleep(0.5)
        if History and self.counts > 0:
            H = HistoryThread(self.HistoryData, 0, self.counts, self.nearOption, self.nextOption)
            H.start()
            # H.join()
            # while temp!=self.counts:
            #     H = HistoryThread(self.HistoryData, temp, self.counts)
            #     temp = self.counts
            #     H.start()
            #     H.join()

        self.getPriceOntime(self.ThreadMonitor)

        self.UpdateFlag = False

        # self.Timer()

        # self.ThreadMonitor.stop()

    def OptionGetFilter(self):
        rd = FrRedis("168.36.1.170")
        dmdi = DailyMdi(rd)
        opkind = self.symbol[:6]

        lst = dmdi.d_icode_opi
        for k, v in lst.items():
            lst[k] = v.__dict__
        lst = pd.DataFrame(lst).transpose()

        a_yymm = dmdi.oplst_a_yymm(opkind)  # > 排序 ['2012', '2101', '2103', '2106']
        lst = lst[(lst.opkind == opkind) & ((lst.yymm == a_yymm[0]) | (lst.yymm == a_yymm[1]))]
        lst.set_index(['yymm', 'icode'], inplace=True, drop=False)
        lst = lst.loc[:, ['cp', 'yymm', 'pxu', 'px']]

        nearOption = lst.loc[a_yymm[0], :]
        nextOption = lst.loc[a_yymm[1], :]

        nearOption.loc[:, 'src'] = 'OP' + opkind + ':#0:' + nearOption.pxu + ':' + nearOption.cp
        self.nearOption = nearOption.sort_values(['cp', 'px', 'icode'])
        self.nearOption = self.nearOption.groupby('cp').apply(lambda x: self.DeltaKGenerate(x))
        # print(self.nearOption)
        self.nearexpire = pd.to_datetime(dmdi.d_opkind_d_yymm_exdate[opkind][a_yymm[0]]) + dt.timedelta(hours=15,
                                                                                                        minutes=30, seconds=0, microseconds=0)
        nextOption.loc[:, 'src'] = 'OP' + opkind + ':#1:' + nextOption.pxu + ':' + nextOption.cp
        self.nextOption = nextOption.sort_values(['cp', 'px', 'icode'])
        self.nextOption = self.nextOption.groupby('cp').apply(lambda x: self.DeltaKGenerate(x))
        # print(self.nextOption)
        self.nextexpire = pd.to_datetime(dmdi.d_opkind_d_yymm_exdate[opkind][a_yymm[1]]) + dt.timedelta(hours=15,
                                                                                                        minutes=30, seconds=0, microseconds=0)

    def timestampinit(self):
        '''
        evaluate the counts starts from 9:30
        by 11:30 would receive 7201 datas and counts = 7201
        stop counting between 11:30-13:00
        by 15:00 would receive 14402 datas and counts = 14402
        this is the terminal value
        ## this index is for filling history data
        '''
        stamp = dt.datetime.now()
        datestamp = dt.datetime(stamp.year, stamp.month, stamp.day, 9, 30)
        n = (stamp - datestamp).total_seconds()
        n = int(n)
        if n <= 0:
            self.counts = 0
        elif 0 < n <= 7200:
            self.counts = n
        elif 7200 < n < 10800 + 1800:
            self.counts = 7201
        elif 10800 + 1800 <= n < 18000 + 1800:
            self.counts = 7201 + n - 10800 - 1800
        else:
            self.counts = 14402

        self.showDate = dt.datetime.now().date()

    def serverConn(self):
        # 连接在用完后要及时断开！！！！
        # 未处理，处理完后这个注释后会更新一行状态！
        # closeEvent
        self.conn = redis.Redis(host='168.36.1.181', port=6379, db=9, password='', charset='gb18030',
                                errors='replace', decode_responses=True, )
        self.conn_r = self.conn.pipeline(transaction=False)

    def getPriceOntime(self, thief: MyThread):
        if thief.countsTag == thief.counts:
            return
        else:
            self.conn_r.hmget(thief.key, self.OPS)
            totalPrice = pd.Series(self.conn_r.execute()[0], index=self.OPSidx). \
                             astype('float').replace(0.0, np.nan) / 10
            # print(totalPrice)
            # print(self.nearOption)
            self.nearOption['currentPrice'] = totalPrice[self.nearOption.index]
            self.nearOption = self.nearOption.groupby('cp').apply(lambda x: self.interpolate(x))
            # print(self.nearOption)

            self.nextOption['currentPrice'] = totalPrice[self.nextOption.index]
            self.nextOption = self.nextOption.groupby('cp').apply(lambda x: self.interpolate(x))
            # print(self.nextOption)
            if self.counts < 14402:
                self.evaluation(self.counts, self.nearOption, self.nextOption)

            thief.followup()
            self.counts = thief.counts

        # print(self.nearOption)

    def interpolate(self, options):
        return options.interpolate(method='linear')

    def DeltaKGenerate(self, options):
        DeltaK = [0] * len(options)
        DeltaK[0] = int(options.iloc[1, 3]) - int(options.iloc[0, 3])
        DeltaK[-1] = int(options.iloc[-1, 3]) - int(options.iloc[-2, 3])
        for i in range(1, (len(options)) - 1):
            DeltaK[i] = (int(options.iloc[i + 1, 3]) - int(options.iloc[i - 1, 3])) / 2

        options['DeltaK'] = DeltaK
        return options

    def evaluation(self, counts, nearOption=None, nextOption=None):
        # if nearOption is None:
        #     nearOption = self.nearOption
        # if nextOption is None:
        #     nextOption = self.nextOption
        # 寻找平价
        T = nearOption.groupby('pxu').apply(lambda x: self.CallMPut(x))
        # print(T)
        # K0near = self.nearOption.reset_index().set_index('pxu', drop=False).loc[T.idxmin(),:]
        K0near = nearOption[nearOption.pxu == T.idxmin()]
        # K0near = T.idxmin()
        Fnear = K0near.set_index('cp')
        Fnear = float(Fnear.px[0]) + np.exp(self.R * self.TminRemainingNear[counts]) * \
                (float(Fnear.loc['C', 'currentPrice']) - float(Fnear.loc['P', 'currentPrice']))
        if abs(Fnear - round(Fnear / 100) * 100) < 1:
            Fnear = round(Fnear / 100) * 100
        K0near = nearOption.px[nearOption.px <= Fnear].max()
        K0near = nearOption[nearOption.px == K0near]

        T = nextOption.groupby('pxu').apply(lambda x: self.CallMPut(x))
        # print(T)
        # self.K0next = self.nextOption.reset_index().set_index('pxu', drop=False).loc[T.idxmin(), :]
        K0next = nextOption[nextOption.pxu == T.idxmin()]
        Fnext = K0next.set_index('cp')
        # print(Fnext)
        Fnext = float(Fnext.px[0]) + np.exp(self.R * self.TminRemainingNext[counts]) * \
                (float(Fnext.loc['C', 'currentPrice']) - float(Fnext.loc['P', 'currentPrice']))
        if abs(Fnext - round(Fnext / 100) * 100) < 1:
            Fnext = round(Fnext / 100) * 100
        K0next = nextOption.px[nextOption.px <= Fnext].max()
        K0next = nextOption[nextOption.px == K0next]

        compOptionsNearP, compOptionsNearC, compOptionsNextP, compOptionsNextC = self.InmoneyOptions(K0near, K0next, nearOption, nextOption)
        # compOptionsNear, compOptionsNext = self.InmoneyOptions(Fnear, Fnext, nearOption, nextOption)

        sigmaNearP = 2 / self.TminRemainingNear[counts] * (compOptionsNearP.DeltaK / compOptionsNearP.px ** 2 *
                                                          compOptionsNearP.currentPrice).sum() * np.exp(
            self.TminRemainingNear[counts] * self.R) - \
                    1 / 2 / self.TminRemainingNear[counts] * (Fnear / K0near.px[0] - 1) ** 2

        sigmaNearC = 2 / self.TminRemainingNear[counts] * (compOptionsNearC.DeltaK / compOptionsNearC.px ** 2 *
                                                          compOptionsNearC.currentPrice).sum() * np.exp(
            self.TminRemainingNear[counts] * self.R) - \
                    1 / 2 / self.TminRemainingNear[counts] * (Fnear / K0near.px[0] - 1) ** 2

        sigmaNextP = 2 / self.TminRemainingNext[counts] * (compOptionsNextP.DeltaK / compOptionsNextP.px ** 2 *
                                                          compOptionsNextP.currentPrice).sum() * np.exp(
            self.TminRemainingNext[counts] * self.R) - \
                    1 / 2 / self.TminRemainingNext[counts] * (Fnext / K0next.px[0] - 1) ** 2

        sigmaNextC = 2 / self.TminRemainingNext[counts] * (compOptionsNextC.DeltaK / compOptionsNextC.px ** 2 *
                                                          compOptionsNextC.currentPrice).sum() * np.exp(
            self.TminRemainingNext[counts] * self.R) - \
                    1 / 2 / self.TminRemainingNext[counts] * (Fnext / K0next.px[0] - 1) ** 2



        NearComponentC = 365 / 30 * self.TminRemainingNear[counts] * sigmaNearC * \
                        (self.TminRemainingNext[counts] - 30 / 365) / (
                                self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        NearComponentP = 365 / 30 * self.TminRemainingNear[counts] * sigmaNearP * \
                         (self.TminRemainingNext[counts] - 30 / 365) / (
                                 self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        NextComponentC = 365 / 30 * self.TminRemainingNext[counts] * sigmaNextC * \
                        (30 / 365 - self.TminRemainingNear[counts]) / (
                                self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        NextComponentP = 365 / 30 * self.TminRemainingNext[counts] * sigmaNextP * \
                        (30 / 365 - self.TminRemainingNear[counts]) / (
                                self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        sigmaOntimeC = 100 * np.sqrt(NearComponentC + NextComponentC)
        sigmaOntimeP = 100 * np.sqrt(NearComponentP + NextComponentP)

        self._Sr_vixOntimeC[counts] = sigmaOntimeC
        self._Sr_vixOntimeP[counts] = sigmaOntimeP
        self._Sr_vixOntime[counts] = sigmaOntimeP+sigmaOntimeC

        self._Sr_vixNearOntimeC[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNear[counts] * sigmaNearC)
        self._Sr_vixNearOntimeP[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNear[counts] * sigmaNearP)
        self._Sr_vixNearOntime[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNear[counts] * (sigmaNearP + sigmaNearC))
        # 近期合约波动率

        self._Sr_vixNextOntimeC[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNext[counts] * sigmaNextC)
        self._Sr_vixNextOntimeP[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNext[counts] * sigmaNextP)
        self._Sr_vixNextOntime[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNext[counts] * (sigmaNextC + sigmaNextP))
        # 远期合约波动率
        # print(counts)
        # print(compOptionsNext)

        # print(self.TminRemainingNear[counts])
        # print(self.TminRemainingNext[counts])

        # print((Fnear / K0near.px[0] - 1)**2/self.TminRemainingNext[counts])
        # tt = (Fnext / K0next.px[0] - 1)**2/self.TminRemainingNext[counts]

        # print(Fnear)
        # print(Fnext)
        # print(K0next.px[0])

        # print(sigmaNear)
        # print(sigmaNext)
        # print(self._Sr_vixOntime[counts])

    def CallMPut(self, options: pd.DataFrame):
        options = options.set_index('cp')
        # print(options)
        return abs(options.loc['C', 'currentPrice'] - options.loc['P', 'currentPrice'])

    def InmoneyOptions(self, K0near, K0next, nearOption, nextOption):
        # def InmoneyOptions(self, Fnear, Fnext, nearOption, nextOption):
        base = K0near.px[0]
        compOptionsNearP = nearOption[(nearOption.px <= base) & (nearOption.cp == 'P')]
        compOptionsNearC = nearOption[(nearOption.px >= base) & (nearOption.cp == 'C')]
        # print(self.compOptionsNear)
        base = K0next.px[0]
        compOptionsNextP = nextOption[(nextOption.px <= base) & (nextOption.cp == 'P')]
        compOptionsNextC = nextOption[(nextOption.px >= base) & (nextOption.cp == 'C')]
        # print(self.compOptionsNext)
        return compOptionsNearP, compOptionsNearC, compOptionsNextP, compOptionsNextC

    def HistoryData(self, startcount, endcount, nearOption, nextOption):
        conn = redis.Redis(host='168.36.1.181', port=6379, db=9, password='', charset='gb18030',
                           errors='replace', decode_responses=True, )
        conn_r = conn.pipeline()

        for counts in np.arange(startcount, endcount):
            key = rKeysList[counts]
            conn_r.hmget(key, self.OPS)

        Hugedf = pd.DataFrame(conn_r.execute(), columns=self.OPSidx). \
                     astype('float').replace(0.0, np.nan).fillna(method='ffill', limit=180, axis=0) / 10

        # print(self.counts)
        # print(Hugedf)

        for counts in np.arange(startcount, endcount):
            # key = rKeysList[counts]
            # conn_r.hmget(key, nearOp)
            # nearOptionPrice = conn_r.execute()[0]
            # self.nearOption['currentPrice'] = pd.Series(nearOptionPrice, dtype='float', index=self.nearOption.index). \
            #                                       replace(0.0, np.nan) / 10
            # print(nearOptionPrice)
            nearOption['currentPrice'] = Hugedf.loc[counts - startcount, nearOption.index]
            # print(self.nearOption)
            nearOption = nearOption.groupby('cp').apply(lambda x: self.interpolate(x))

            # print(self.nearOption)

            # conn_r.hmget(key, nextOp)
            # nextOptionPrice = conn_r.execute()[0]

            # self.nextOption['currentPrice'] = pd.Series(nextOptionPrice, dtype='float', index=self.nextOption.index). \
            #                                       replace(0.0, np.nan) / 10
            nextOption['currentPrice'] = Hugedf.loc[counts - startcount, self.nextOption.index]
            # print(self.nextOption)
            nextOption = nextOption.groupby('cp').apply(lambda x: self.interpolate(x))

            self.evaluation(counts, nearOption, nextOption)

        conn_r.close()
        conn.close()

    # def Timer(self):
    #     # connect to self.getPriceOntime()
    #     pass


class backstage():
    def __init__(self, q, symbol, History=True):

        self.q = q
        self.symbol = symbol
        self.K = VIXontime(symbol, History)
        self.counts = self.K.counts

        self.publisher = redis.Redis(host='168.36.1.181', db=9, port=6379, password='', charset='gb18030',
                                     errors='replace', decode_responses=True, )

        self.timer_init()

    def timer_init(self):
        self.timer = threading.Timer(0, self.iterator)
        self.timer.start()

    def iterator(self):
        self.update_plot_ontime()
        if self.counts > 14401:
            self.timer.cancel()
            self.q.put('out')
            sys.exit()
        wait = self.sync()
        self.timer = threading.Timer(wait, self.iterator)
        print(dt.datetime.now())
        self.timer.start()

    def sync(self):
        return ((dt.datetime.now().replace(microsecond=0)+dt.timedelta(seconds=1)) -
                dt.datetime.now()).total_seconds()

        # self.timer = threading.Timer(0.5, self.iterator)
        # self.timer.start()

    def update_plot_ontime(self):
        self.K.getPriceOntime(self.K.ThreadMonitor)
        if 0 < self.K.counts < 14402 and self.counts != self.K.counts:

            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#01:ZH',
                                value=self.K._Sr_vixOntime[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#0:ZH',
                                value=self.K._Sr_vixNearOntime[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#1:ZH',
                                value=self.K._Sr_vixNextOntime[self.K.counts - 1])

            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#01:C',
                                value=self.K._Sr_vixOntimeC[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#0:C',
                                value=self.K._Sr_vixNearOntimeC[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#1:C',
                                value=self.K._Sr_vixNextOntimeC[self.K.counts - 1])

            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#01:P',
                                value=self.K._Sr_vixOntimeP[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#0:P',
                                value=self.K._Sr_vixNearOntimeP[self.K.counts - 1])
            self.publisher.hset(name=rKeysList[self.counts], key=self.symbol[:6] + ':#1:P',
                                value=self.K._Sr_vixNextOntimeP[self.K.counts - 1])

            self.q.put(rKeysList[self.counts])

            self.counts = self.K.counts
        else:
            self.counts = self.K.counts


def main():
    # global symbol
    # K = VIXontime()
    # T = K.nearOption.groupby('pxu').apply(lambda x: tt(x))
    # print(T)
    # 修改标的
    if len(sys.argv) > 1:
        symbolList = sys.argv[1].split(',')
    else:
        sys.exit()
    Flag = multiprocessing.Queue(100)
    publisher = redis.Redis(host='168.36.1.181', db=9, port=6379, password='', charset='gb18030',
                                     errors='replace', decode_responses=True, )
    tasks = [multiprocessing.Process(target=backstage, args=(Flag, symbol, False)) for symbol in symbolList]
    L = len(symbolList)

    D = defaultdict(lambda: 0)

    for task in tasks:
        task.start()
    while True:
        try:
            F = Flag.get()
            D[F] = D[F]+1
            if D[F] == L:
                if F=='out':
                    break
                else:
                    publisher.publish(channel='V:', message=F)
                    del D[F]
        except:
            publisher.close()
            sys.exit()


if __name__ == '__main__':
    main()
