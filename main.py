import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import altair as alt
import pymysql
import streamlit_authenticator as stauth
from PIL import Image
import yaml
import bank
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Bank - Customer - Segmentation",
    page_icon="🎉",layout="wide")

session_state = {}

def login():
    st.title("Đăng nhập")
    
    if 'logged_in' not in session_state or not session_state['logged_in']:
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        
        if st.button("Đăng nhập"):
            if username == "admin" and password == "123":
                session_state['logged_in'] = True
                session_state['username'] = username
                main_page()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không chính xác!")
    else:
        main_page()

def main_page():
    if st.button("Đăng xuất"):
        session_state['logged_in'] = False
        st.info("Bạn đã đăng xuất thành công!")
        login()

    st.sidebar.image("D:\TẢI XUÔNG_2\VanAnh Bank customer Churn\Bank_customer_segment-master\Image\dss.png")
    with st.sidebar:
        selected = option_menu("Main Menu", ["Dữ liệu", 'Phân cụm'])
    if selected == "Dữ liệu":
        st.subheader("Dữ liệu")
        tab1, tab2 = st.tabs(["Tổng quan", "Đồ thị"])
        with tab1:
            uploaded_file = st.file_uploader("Upload file CSV", type="csv")
            if uploaded_file is not None:
                tab1.subheader("Tổng quan")
                data = pd.read_csv(uploaded_file)
                st.dataframe(data)
                st.write(f"Số lượng khách hàng: {bank.creditcard_df.shape}")
                if data is not None:
                    st.subheader("Tìm kiếm")
                    id_tk = st.text_input("Nhập ID khách hàng:","")
                    try:
                        id_tk=int(bank.creditcard_df['CUST_ID'])
                    except:
                        pass
                    st.write(st.write(data[data.CUST_ID==id_tk]))
    if selected == 'Phân cụm':
        st.subheader("Phân cụm dữ liệu")
        # Xác định giá trị CUSTID đầu vào
        cust_id = st.text_input("Nhập giá trị CUSTID")

# # Tìm cụm tương ứng trong pca_df
#     if cust_id:
#         try:
#             bank.creditcard_df_cluster['CUST_ID'] = bank.creditcard_df['CUST_ID'].astype(int)

#             index = bank.creditcard_df_cluster[bank.creditcard_df_cluster['CUST_ID'] == int(cust_id)].index[0]
#             cluster = bank.creditcard_df_cluster.loc[index, 'cluster']
#             st.write(f"Kết quả phân cụm tương ứng với CUSTID {cust_id}: {cluster}")
#         except IndexError:
#             st.write("CUSTID không tồn tại trong dữ liệu")

                
        st.subheader("Vẽ biểu đồ cột từng cụm")
        for feature in bank.creditcard_df.columns:
            fig=plt.figure(figsize=(35, 5))
            for cluster in range(5):
                plt.subplot(1, 5, cluster+1)
                cluster_data = bank.creditcard_df_cluster[bank.creditcard_df_cluster['cluster'] == cluster]
                cluster_data[feature].hist(bins=20)
                plt.title('Cluster {} - {}'.format(cluster, feature))
            st.pyplot(fig)
        fig11=plt.figure(figsize=(10,10))
        ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = bank.pca_df, palette =['red','green','blue','pink','yellow'])
        st.pyplot(fig11)
 
  
  
        
        

        
login()
main_page()


 