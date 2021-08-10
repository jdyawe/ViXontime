import json
import urllib.request
import pandas as pd
import time
import matplotlib.pyplot as plt

def get_data(url_temp,retries=3):
    try:
        f = urllib.request.urlopen(url_temp, timeout=1)
        data = f.read()
    except Exception as e:
        print(str(e), url_temp)
        if retries > 0:
            time.sleep(0.5)
            return get_data(url_temp, retries-1)
        else:
            print('GET Failed', url_temp)
            return -1
    return data

url = 'http://yunhq.sse.com.cn:32041/v1/csip/dayk/000188?callback=test&begin=1&end=-1&select=date%2Copen%2Chigh%2Clow%2Cclose&_=1492899476267'

return_str = get_data(url)
return_str = return_str[5:-1]
json_obj = json.loads(return_str)
kline = pd.DataFrame(json_obj['kline'], columns=['date', 'open', 'high', 'low', 'close'])
# print(kline)

# plt.plot(kline.close)
# plt.show()
