# import requests
# import bs4
# import pandas as pd
# import time
# import json
# import os
# from datetime import datetime
#
#
# #
# # # hahow crawl web api request  請改日期
# url = "https://api.hahow.in/api/products/search?category=COURSE&filter=INCUBATING&limit=24&page=0&sort=INCUBATE_TIME"
#
# headers = {
#        'User-Agent':
#    'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36'
# }
#
# res = requests.get(url, headers = headers)
#
# if res.status_code == 200:
#     data = res.json()
#     # print(data['data']['courseData']['products'])
#     products = data['data']['courseData']['products']
#     course_list = []
#     for product in products:
#         course_data = [
#              product['title'],
#              product['preOrderedPrice'],
#              product['incubateTime'],
#              product['proposalDueTime'],
#              product['numSoldTickets']
#         ]
#
#         course_list.append( course_data)
#     df = pd.DataFrame(course_list, columns=['title', 'preOrderedPrice', 'startdate', 'enddate', 'number'])
#     df.to_excel('2024_03_04_hahow.xlsx', index=False, engine='openpyxl')
#     print('SAVE')
# else:
#     print("fail")
import requests
import pandas as pd
import time
from datetime import datetime
import schedule


def fetch_hahow_data():
    url = "https://api.hahow.in/api/products/search?category=COURSE&filter=INCUBATING&limit=24&page=0&sort=INCUBATE_TIME"

    headers = {
        'User-Agent':
            'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36'
    }

    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        data = res.json()
        products = data['data']['courseData']['products']
        course_list = []
        for product in products:
            course_data = [
                product['title'],
                product['preOrderedPrice'],
                product['incubateTime'],
                product['proposalDueTime'],
                product['numSoldTickets']
            ]

            course_list.append(course_data)

        # 獲取當前日期作為檔案名的一部分
        today_date = datetime.now().strftime("%Y_%m_%d")
        filename = f'{today_date}_hahow.xlsx'

        df = pd.DataFrame(course_list, columns=['title', 'preOrderedPrice', 'startdate', 'enddate', 'number'])
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f'SAVE: {filename}')
    else:
        print("Request failed")


# 設定排程：每天執行一次
schedule.every().day.at("00:00").do(fetch_hahow_data)

# 保持程式運行，等待排程任務執行
while True:
    schedule.run_pending()
    time.sleep(1)






