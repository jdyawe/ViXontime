import jqdatasdk as jds
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

from ut_redis import FrRedis
from ut_dmdi import DailyMdi

import multiprocessing

import seaborn as sns

import scipy
import sklearn as sl

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from sublist import rKeysList

# eft复权后取平价合约和直接利用沽购合约价差最小取平价合约的结果不一定相同！！！

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
    def __init__(self, History=True):
        self._Sr_vixOntime = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNearOntime = pd.Series([np.nan] * 14402, dtype='float64')
        self._Sr_vixNextOntime = pd.Series([np.nan] * 14402, dtype='float64')

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
        if History and self.counts>0:
            H = HistoryThread(self.HistoryData, 0, self.counts, self.nearOption, self.nextOption)
            H.start()
            # H.join()
            # while temp!=self.counts:
            #     H = HistoryThread(self.HistoryData, temp, self.counts)
            #     temp = self.counts
            #     H.start()
            #     H.join()



        self.getPriceOntime(self.ThreadMonitor)

        # self.Timer()

        # self.ThreadMonitor.stop()

    def OptionGetFilter(self):
        rd = FrRedis("168.36.1.170")
        dmdi = DailyMdi(rd)
        opkind = symbol[:6]

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
                                                                                                        minutes=30)
        nextOption.loc[:, 'src'] = 'OP' + opkind + ':#1:' + nextOption.pxu + ':' + nextOption.cp
        self.nextOption = nextOption.sort_values(['cp', 'px', 'icode'])
        self.nextOption = self.nextOption.groupby('cp').apply(lambda x: self.DeltaKGenerate(x))
        # print(self.nextOption)
        self.nextexpire = pd.to_datetime(dmdi.d_opkind_d_yymm_exdate[opkind][a_yymm[1]]) + dt.timedelta(hours=15,
                                                                                                        minutes=30)

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
        if abs(Fnext-round(Fnext/100)*100)<1:
            Fnext = round(Fnext/100)*100
        K0next = nextOption.px[nextOption.px <= Fnext].max()
        K0next = nextOption[nextOption.px == K0next]

        compOptionsNear, compOptionsNext = self.InmoneyOptions(K0near, K0next, nearOption, nextOption)
        # compOptionsNear, compOptionsNext = self.InmoneyOptions(Fnear, Fnext, nearOption, nextOption)
        sigmaNear = 2 / self.TminRemainingNear[counts] * (compOptionsNear.DeltaK / compOptionsNear.px ** 2 *
                    compOptionsNear.currentPrice).sum() * np.exp(self.TminRemainingNear[counts] * self.R) - \
                         1 / self.TminRemainingNear[counts] * (Fnear / K0near.px[0] - 1) ** 2

        sigmaNext = 2 / self.TminRemainingNext[counts] * (compOptionsNext.DeltaK / compOptionsNext.px ** 2 *
                    compOptionsNext.currentPrice).sum() * np.exp(self.TminRemainingNext[counts] * self.R) - \
                         1 / self.TminRemainingNext[counts] * (Fnext / K0next.px[0] - 1) ** 2

        NearComponent = 365 / 30 * self.TminRemainingNear[counts] * sigmaNear * \
                        (self.TminRemainingNext[counts] - 30 / 365) / (
                                    self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        NextComponent = 365 / 30 * self.TminRemainingNext[counts] * sigmaNext * \
                        (30 / 365 - self.TminRemainingNear[counts]) / (
                                    self.TminRemainingNext[counts] - self.TminRemainingNear[counts])
        sigmaOntime = 100 * np.sqrt(NearComponent + NextComponent)

        self._Sr_vixOntime[counts] = sigmaOntime
        self._Sr_vixNearOntime[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNear[counts] * sigmaNear)
        #近期合约波动率
        self._Sr_vixNextOntime[counts] = 100 * np.sqrt(365 / 30 * self.TminRemainingNext[counts] * sigmaNext)
        #远期合约波动率

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
        compOptionsNear = pd.concat([nearOption[(nearOption.px <= base) & (nearOption.cp == 'P')],
                                          nearOption[(nearOption.px >= base) & (nearOption.cp == 'C')]])
        # print(self.compOptionsNear)
        base = K0next.px[0]
        compOptionsNext = pd.concat([nextOption[(nextOption.px <= base) & (nextOption.cp == 'P')],
                                          nextOption[(nextOption.px >= base) & (nextOption.cp == 'C')]])
        # print(self.compOptionsNext)
        return compOptionsNear, compOptionsNext

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


class MainUI(QtWidgets.QMainWindow):
    def __init__(self, History=True):
        super(MainUI, self).__init__()

        self.K = VIXontime(History)

        self.publisher = redis.Redis(host='168.36.1.181', db=5, port=6379, password='',charset='gb18030',
                           errors='replace', decode_responses=True,)

        self.setWindowTitle('VIXonTime' + symbol)
        # main window create

        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.main_layout.setSpacing(0)

        self.Plotting_Window()
        self.main_layout.addWidget(self.plot_window, stretch=1)

        self.timer_init()

    def Plotting_Window(self):
        self.plot_window = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QGridLayout()
        self.plot_window.setLayout(self.plot_layout)

        TimeLabel = \
            {
                0: "09:30:00",
                1800: "10:00:00",
                3600: "10:30:00",
                7200: "11:30:00/13:00:00",
                7201 + 3600: "14:00:00",
                7201 + 5400: "14:30:00",
                7201 + 7200: "15:00:00",
            }

        # 滚轮与右键控件
        class ModifiedViewBox(pg.ViewBox):
            def __init__(self):
                super(ModifiedViewBox, self).__init__()

            def mouseDragEvent(self, ev, axis=None):
                ev.ignore()

            def wheelEvent(self, ev, axis=None):
                ev.ignore()

        self.vb = ModifiedViewBox()
        self.vb.setAutoVisible(y=True)

        # stringXaxis = pg.AxisItem(orientation='bottom')
        # stringXaxis.setTicks(TimeLabel.items())

        self.Canvas = pg.PlotWidget(viewBox=self.vb)
        # self.Canvas = pg.PlotWidget()
        self.Canvas.setBackground('w')
        self.Canvas.getAxis('bottom').setTicks([TimeLabel.items()])
        self.Canvas.showGrid(x=True, y=True)
        self.Canvas.setXRange(0, 14401, padding=0.01)
        # self.Canvas.setYRange(padding=self.Canvas.)

        self.plot_layout.addWidget(self.Canvas)
        # self.plot_layout.setOriginCorner()
        self.plot_layout.setSpacing(0)

        self.presentBox = pg.TextItem()
        # self.Canvas.addItem(self.presentBox)

        # adding signals
        self.LinesPlot = {}
        color = [255, 0, 0]
        self.LinesPlot['pen'] = pg.mkPen(QtGui.QColor(color[0], color[1], color[2]),
                                         width=0.25)

        self.LinesPlot['vix'] = self.Canvas.plot()
        self.LinesPlot['vix'].setData([], [], pen=self.LinesPlot['pen'])

    def timer_init(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.update_plot_ontime())
        self.timer.start(500)

    def update_plot_ontime(self):
        self.K.getPriceOntime(self.K.ThreadMonitor)
        if 0 <= self.K.counts < 14402:
            self.publisher.hset(name='V'+rKeysList[self.K.counts][1:], key='ZH', value=self.K._Sr_vixOntime[self.K.counts-1])
            self.publisher.publish(channel='V:ZH', message=self.K._Sr_vixOntime[self.K.counts])
        self.LinesPlot['vix'].setData(np.arange(self.K.counts), self.K._Sr_vixOntime[:self.K.counts])

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.K.conn_r.close()
        self.K.conn.close()
        self.publisher.close()

# def tt(x):
#     x.set_index('cp', inplace=True)
#     return int(x.loc['C', 'px']) - int(x.loc['P', 'px'])

def main():
    # K = VIXontime()
    # T = K.nearOption.groupby('pxu').apply(lambda x: tt(x))
    # print(T)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUI(True)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
