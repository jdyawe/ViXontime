from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import sys, os, shutil, re
import traceback
import psutil
import redis
import time, datetime
import yaml
from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator, QRegExpValidator
import numpy as np
import pandas as pd
from functools import partial
from operator import itemgetter

import jqdatasdk as jds

from multiprocessing import Process, Value  # 导入multiprocessing模块，然后导入Process这个类

import threading
from time import sleep, ctime

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.set_option('mode.chained_assignment', None)

symbol = "510050.XSHG"
tdy = datetime.datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)


class JQVIXonTime:
    def __init__(self, underlying_asset: str):
        self.ID = '18810592263'
        self.pwd = '592263'
        jds.auth(self.ID, self.pwd)
        self.R = 0.02  # 无风险利率 约为1年期中国国债

        Xrange1 = pd.date_range(start=tdy.replace(hour=9, minute=30, second=0, microsecond=0),
                                end=tdy.replace(hour=11, minute=30, second=0, microsecond=0),
                                freq='T')
        Xrange2 = pd.date_range(start=tdy.replace(hour=13, minute=0, second=0, microsecond=0),
                                end=tdy.replace(hour=15, minute=0, second=0, microsecond=0),
                                freq='T')
        self.Xrange = Xrange1.append(Xrange2)

        self.fields = ['code', 'name', 'exercise_price', 'contract_type', 'list_date']
        self.chosen = ['pre_settle', 'settle_price', 'change_pct_settle', 'pre_close', 'open', 'close', 'volume', ]

        # 筛选历史合约信息
        q = jds.query(jds.opt.OPT_DAILY_PREOPEN) \
            .filter((jds.opt.OPT_DAILY_PREOPEN.underlying_symbol == underlying_asset)
                    & (jds.opt.OPT_DAILY_PREOPEN.date == tdy.date())
                    & (jds.opt.OPT_DAILY_PREOPEN.expire_date >= tdy)
                    & (jds.opt.OPT_DAILY_PREOPEN.expire_date <= tdy + datetime.timedelta(days=70)))
        self.options = jds.opt.run_query(q).loc[:, ['code', 'contract_type', 'exercise_price', 'expire_date']]

        self.expireDates = sorted(self.options.expire_date.drop_duplicates().tolist())[:2]
        self.NearOpExpireDate = pd.to_datetime(self.expireDates[0]).replace(hour=15)
        self.NextOpExpireDate = pd.to_datetime(self.expireDates[1]).replace(hour=15)
        self.options = self.options[self.options.expire_date <= self.expireDates[1]]
        self.options['close'] = np.nan
        self.options = self.options.groupby(['expire_date', 'contract_type']).apply(lambda x: self.DeltaK(x))
        self.options.set_index('code', inplace=True)

        self.VIX_Ontime = pd.Series(np.nan*len(self.Xrange), dtype='float', index=self.Xrange)
        self.VIX_TICK_DATA = pd.Series(dtype='float')

    def getPriceOntime(self):
        self.price = jds.get_bars(security=self.options.index.to_list(), count=1, unit='1m', fields=['close'],
                                  end_dt=(datetime.datetime.now()+datetime.timedelta(minutes=1)).replace(second=0),
                                  include_now=True).reset_index(level=0).set_index('level_0')
        self.ticktime = datetime.datetime.now().replace(microsecond=0)
        self.timestamp = (datetime.datetime.now()+datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
        self.options['close'] = self.price
        # self.options = self.options.reset_index().set_index(['contract_type', 'exercise_price'])

    def DeltaK(self, x):
        ep = x.sort_values(by='exercise_price').exercise_price
        dif = ep.diff(2).shift(-1) / 2
        dif.iloc[0] = ep.iloc[1] - ep.iloc[0]
        dif.iloc[-1] = ep.iloc[-1] - ep.iloc[-2]
        x['diff'] = dif
        return x.sort_values(by=['expire_date', 'contract_type', 'exercise_price'])

    def evaluation(self):
        self.getPriceOntime()
        # 得到平价期权
        temp = (self.options.reset_index().set_index(['contract_type', 'exercise_price'])). \
            groupby('expire_date').apply(
            lambda x: pd.concat({'C': x.loc['CO', 'close'], 'P': x.loc['PO', 'close']}, axis=1, sort=True))
        temp.reset_index(inplace=True)

        exP = temp.groupby('expire_date').apply(lambda x: self.Inmoney(x)).set_index('expire_date')

        # 算出F-合成期货价格
        # 得到K0
        Tnear = (self.NearOpExpireDate - datetime.datetime.now()).total_seconds() // 60 / 365 / 1440
        Tnext = (self.NextOpExpireDate - datetime.datetime.now()).total_seconds() // 60 / 365 / 1440

        Fnear = exP.loc[self.expireDates[0], 'exercise_price'] + np.exp(self.R * Tnear) * exP.loc[
            self.expireDates[0], 'diff']
        Fnext = exP.loc[self.expireDates[1], 'exercise_price'] + np.exp(self.R * Tnext) * exP.loc[
            self.expireDates[1], 'diff']

        K0near = self.options[self.options.expire_date == self.expireDates[0]].reset_index().set_index('exercise_price',
                                                                                                       drop=False).exercise_price - Fnear
        K0near = (K0near[K0near <= 0]).idxmax()

        K0next = self.options[self.options.expire_date == self.expireDates[1]].reset_index().set_index('exercise_price',
                                                                                                       drop=False).exercise_price - Fnext
        K0next = (K0next[K0next <= 0]).idxmax()

        K0 = {self.expireDates[0]: K0near, self.expireDates[1]: K0next}

        # 算"实值"合约列表

        realOption = (self.options.reset_index().set_index(['contract_type', 'exercise_price'])).\
            groupby('expire_date').apply(lambda x: self.InmoneyOptions(x, K0))

        realOption.reset_index(inplace=True, level=[1, 2])
        realOption.reset_index(inplace=True, drop=True)
        nearOption = realOption[realOption.expire_date == self.expireDates[0]]
        nextOption = realOption[realOption.expire_date == self.expireDates[1]]

        if Tnear != 0:
            sigmanear = (nearOption.loc[:, 'diff'] / (nearOption.exercise_price ** 2) * nearOption.close).sum()
            sigmanear = 2 / Tnear * np.exp(self.R * Tnear) * sigmanear - 1 / Tnear * (Fnear / K0near - 1) ** 2
        else:
            sigmanear = 0

        sigmanext = (nextOption.loc[:, 'diff'] / (nextOption.exercise_price ** 2) * nextOption.close).sum()
        sigmanext = 2 / Tnext * np.exp(self.R * Tnext) * sigmanext - 1 / Tnext * (Fnext / K0next - 1) ** 2

        sigma = Tnear * sigmanear * (Tnext - 30 / 365) / (Tnext - Tnear) + Tnext * sigmanext * (30 / 365 - Tnear) / (
                Tnext - Tnear)
        sigma = sigma * 365 / 30
        sigma = 100 * np.sqrt(sigma)
        self.VIX_Ontime[self.timestamp] = sigma
        self.VIX_TICK_DATA[self.ticktime] = sigma
        print(self.timestamp, sigma)


    def Inmoney(self, x):
        x['diff'] = x.C - x.P
        xt = (x.loc[:, 'C'] - x.loc[:, 'P']).abs()
        xt.index = x.exercise_price.values
        return x[x.exercise_price == xt.idxmin()]

    def InmoneyOptions(self, x, K0):
        templst = x.index.tolist()
        d = x.expire_date.drop_duplicates().tolist()[0]
        templst = [(a, b) for a, b in templst if (a == 'CO' and b >= K0[d]) or (a == 'PO' and b <= K0[d])]
        return x.loc[templst, :]


class MainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()

        self.K = JQVIXonTime(symbol)

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
                30: "10:00:00",
                60: "10:30:00",
                120: "11:30:00/13:00:00",
                180: "14:00:00",
                210: "14:30:00",
                240: "15:00:00",
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
        self.Canvas.setXRange(0, 240, padding=0.01)
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
                                         width=1)

        self.LinesPlot['vix'] = self.Canvas.plot()
        self.LinesPlot['vix'].setData([], [], pen=self.LinesPlot['pen'])

    def timer_init(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.update_plot_ontime())
        self.timer.start(1000)

    def update_plot_ontime(self):
        self.K.evaluation()
        if self.K.timestamp <= self.K.Xrange[-1]:
            temp = sum(self.K.Xrange<=self.K.timestamp)+1
            self.LinesPlot['vix'].setData(range(temp),
                                        self.K.VIX_Ontime.iloc[:temp])

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        file = datetime.datetime.now().date().strftime("%Y%m%d")+'VIXTICK.csv'
        with open(file, 'w') as f:
            self.K.VIX_TICK_DATA.to_csv(f)


# def tt(x):
#     x.set_index('cp', inplace=True)
#     return int(x.loc['C', 'px']) - int(x.loc['P', 'px'])

def main():
    # K = VIXontime()
    # T = K.nearOption.groupby('pxu').apply(lambda x: tt(x))
    # print(T)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


