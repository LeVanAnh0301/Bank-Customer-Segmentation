import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

df= pd.read_csv('D:\TẢI XUÔNG_2\VanAnh Bank customer Churn\Bank_customer_segment-master\marketing_data.csv')
# Tạo cột "Date" trong khoảng thời gian 6 tháng từ ngày 1/1/2021
# Thêm cột "AccountID" với giá trị từ 1 đến 8950
df['AccountID'] = np.arange(1, 8951)
# print(df.head())
print(df.tail())
df.to_csv('dataf.csv', index=False)
