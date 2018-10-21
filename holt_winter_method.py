# -*- coding:utf-8 -*-

import pandas as pd
from pylab import *
import numpy as np
from datetime import timedelta
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def holt_winter_method(log_data, alpha=0.8, beta=0.05, gammar=0.15, confidence_ratio=0.2, period_k=7):
    '''
    :param log_data: 记录每天的登录登出次数
    :param alpha: 表示残差数据的平滑系数
    :param beta:  表示趋势数据的平滑系数
    :param gammar:  表示周期性数据的平滑系数
    :param period_k: 表示周期
    :return: 原始数据, 预测数据, 置信下限, 置信上限, 均方误差
    '''
    log_data.index = pd.to_datetime(log_data.index)
    log_data_list = []
    for j in xrange(len(log_data.index)):
        data = log_data.loc[log_data.index[j], 'log_count']
        log_data_list.append(float(data))

    original_data = log_data_list
    residual_data = []; trend_data = []; period_data = []
    predicted_data = []
    future_predict = []
    MSE = 0        #均方误差
    record = 0
    for j in xrange(len(log_data)):
        record += 1
        if j in np.linspace(0, period_k - 1, period_k):
            residual_data.append(0)
            trend_data.append(0)
            period_data.append(0)
            predicted_data.append(original_data[j])
        else:
           residual_data.append(alpha*(original_data[j] - period_data[j - period_k]) + \
                               (1 - alpha)*(residual_data[j - 1] + trend_data[j - 1]))     #残差数据
           trend_data.append(beta*(residual_data[j] - residual_data[j - 1]) +(1 - beta)*trend_data[j - 1])   #趋势数据
           period_data.append(gammar*(original_data[j] - residual_data[j]) + (1 - gammar)*period_data[j - period_k])  #周期性数据
           predicted_data.append(residual_data[j] + trend_data[j] + period_data[j - period_k + 1])    #预测数据
           # lower_limit = np.multiply(predicted_data, np.ones(len(predicted_data)) * (1 - confidence_ratio))      #置信下限
           # upper_limit = np.multiply(predicted_data, np.ones(len(predicted_data)) * (1 + confidence_ratio))      #置信上限
           MSE = (int(predicted_data[j]) - int(log_data_list[j])) ** 2 + MSE     #均方误差
           if record == len(log_data):
               for count_h in range(1, 8):           #预测未来七天的数据
                    future_predict.append(residual_data[j] + count_h*trend_data[j] + period_data[j - period_k + 1 + (count_h - 1) % period_k])

    MSE = MSE / len(log_data)
    # return log_data_list, predicted_data, lower_limit, upper_limit, future_predict, MSE
    return log_data, predicted_data, future_predict, MSE


def plot_data(log_data, predicted_data, future_predict):
    # lower_limit = np.multiply(predicted_data, np.ones(len(predicted_data))*0.8)      #置信下限
    # upper_limit = np.multiply(predicted_data, np.ones(len(predicted_data))*1.2)      #置信上限
    ax = subplot(1, 1, 1)
    ymajorLocator = MultipleLocator(100)            # 将y轴主刻度标签设置为100的倍数
    ax.yaxis.set_major_locator(ymajorLocator)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    log_datetime = []
    for js_num in xrange(len(log_data.index)):
        log_datetime.append(log_data.index[js_num])

    future_date = []
    for count_h in xrange(1, 8):  # 预测未来七天的数据
        predict_date = log_datetime[-1] + timedelta(count_h)
        future_date.append(predict_date.strftime('%Y/%m/%d'))

    # print "future_date:", future_date
    plt.plot(log_datetime, log_data, 'r-o', label=u'actual_value')
    plt.plot(log_datetime, predicted_data, 'g-', label=u'predicted_value3')
    # plt.plot(log_datetime, lower_limit, 'b-', label=u'lower_limit')
    # plt.plot(log_datetime, upper_limit, 'y-', label=u'upper_limit')
    print future_date
    print future_predict
    future_predict = [predict_value if predict_value > 0 else 0 for predict_value in future_predict]
    plt.plot(future_date, future_predict, label=u'future_predict')
    plt.title(u'actual_and_predict_value')
    plt.xlabel(u'date')
    plt.ylabel(u'values')
    plt.ylim([0, 2000])
    plt.legend((u'actual_value', u'predicted_value3', u'future_predict'))
    # plt.legend((u'actual_value', u'predicted_value3', u'lower_limit', u'upper_limit'))
    plt.show()
    return future_date,future_predict


if __name__ == '__main__':
    log_file_addr = u'log_date_data.xlsx'
    log_data = pd.read_excel(log_file_addr, index_col=u'log_date', sheetname=u'log_count', \
                            dtype={u'log_count': float}, encoding='gb18030')

    log_data, predicted_data, future_predict, MSE = holt_winter_method(log_data)  #依次输出待分析的数据、预测数据、置信下限、置信上限、均方误差
    print MSE
    future_date, future_predict = plot_data(log_data, predicted_data, future_predict)
    print future_date, future_predict

