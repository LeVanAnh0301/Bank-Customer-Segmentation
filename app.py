import streamlit as st
import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as ex
import plotly.offline as pyo
from datetime import datetime
pyo.init_notebook_mode()
sns.set_style('darkgrid')
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
import streamlit as st
from PIL import Image
import altair as alt
import pymysql
import streamlit_authenticator as stauth

import yaml

from pathlib import Path
st.title("Bank Customer Segment")
# Khi người dùng đã đăng nhập thành công, biến flag_login sẽ được đặt thành True
# flag_login = False

# # Dữ liệu người dùng
# user_credentials = {
#     "vananh": "123",
#     "VanAnh": "456",
#     # Thêm các tên người dùng và mật khẩu khác nếu cần thiết
# }

# # Hiển thị giao diện đăng nhập
# username = st.text_input("Tên người dùng:")
# password = st.text_input("Mật khẩu:", type="password")

# # Kiểm tra xác thực
# if st.button("Đăng nhập"):
#     if username in user_credentials:
#         if user_credentials[username] == password:
#             st.success("Đăng nhập thành công!")
#             flag_login = True
#             # Tiếp tục với chức năng chính của ứng dụng
#         else:
#             st.error("Mật khẩu không đúng!")
#     else:
#         st.error("Tên người dùng không tồn tại!")


# st.set_page_config(
#     page_title="Bank Customer Segmentation",
#     page_icon="🦈",layout="wide")

# names = ["Huong", "Huong1"]
# usernames = ["huong", "huong1"]
# passwords=['abc123','def456']
# hashed_passwords = stauth.Hasher(passwords).generate()
# file_path = "hashed_pw.pkl"
    
# with open(file_path,"wb") as file:
#     pickle.dump(hashed_passwords,file)
# # load hashed passwords
# # credentials = {"usernames":{}}

# with  open(file_path,"rb") as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#     "sales_dashboard", "abcdef", cookie_expiry_days=30)
# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status == False:
#     st.error("Username/password is incorrect")
# if authentication_status == None:
#     st.warning("Please enter your username and password")
# # if authentication_status == True:
# #     st.error("Username/password is correct")

# if authentication_status:
#     authenticator.logout("Logout", "sidebar")
#     st.sidebar.title(f"Welcome {name}!")
# if flag_login:
st.sidebar.title(f"Welcome Vân Anh!")
with st.sidebar:
    selected = option_menu("Main Menu", ["Dữ liệu", 'Phân cụm'])
if selected == "Dữ liệu":
    st.subheader("Dữ liệu")
    tab1, tab2 = st.tabs(["Tổng quan", "Đồ thị"])
    with tab1:


        uploaded_file = st.file_uploader("Upload File")
        if uploaded_file is not None:
            tab1.subheader("Tổng quan")
            data = pd.read_csv(uploaded_file)
            with open('data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            st.dataframe(data)
            custom_num =  len(data["CUST_ID"])
            st.write(f"Số lượng khách hàng: {custom_num}")
            if data is not None:
                st.subheader("Tìm kiếm")
                id_tk = st.text_input("Nhập ID khách hàng:","")
                try:
                    id_tk=int(id_tk)
                except:
                    pass
                st.write(st.write(data[data.CUST_ID==id_tk]))
    
    # with tab2: 
    #     with open('data.pkl', 'rb') as f:
    #         data=pickle.load(f)
            
    #     if data is not None:                    
    #         labels = "F", "M"
    #         size = 0.5
    #         fig1,ax=plt.subplots(figsize=(12, 12))
    #         wedges, texts, autotexts = plt.pie([data["Gender"].value_counts()[0],
    #                                             data["Gender"].value_counts()[1],

    #                                             ],

    #                                         explode=(0, 0),
    #                                         textprops=dict(size=20, color="white"),
    #                                         autopct="%.2f%%",
    #                                         pctdistance=0.72,
    #                                         radius=.9,
    #                                         colors=["#682F2F", "#F3AB60"],
    #                                         shadow=True,
    #                                         wedgeprops=dict(width=size, edgecolor="black",
    #                                                         linewidth=4),
    #                                         startangle=20)

    #         plt.legend(wedges, labels, title="Giới tính", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
    #                 edgecolor="black")
    #         st.pyplot(fig1)
            
    #         labels = "Blue", "Silver","Gold","Platinum"
    #         size = 0.5
    #         fig2,axx=plt.subplots(figsize=(12, 12))
    #         wedges, texts, autotexts = plt.pie([data["Card_Category"].value_counts()[0],
    #                                             data["Card_Category"].value_counts()[1],
    #                                             data["Card_Category"].value_counts()[2],
    #                                             data["Card_Category"].value_counts()[3],

    #                                             ],

    #                                         explode=(0, 0,0,0),
    #                                         textprops=dict(size=20, color="white"),
    #                                         autopct="%.2f%%",
    #                                         pctdistance=0.72,
    #                                         radius=.9,
    #                                         colors=["#682F2F", "#F3AB60", "#9F8A78","#EED5B7"],
    #                                         shadow=True,
    #                                         wedgeprops=dict(width=size, edgecolor="black",
    #                                                         linewidth=4),
    #                                         startangle=20)

    #         plt.legend(wedges, labels, title="Các loại thẻ", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
    #                 edgecolor="black")
    #         st.pyplot(fig2)

    #         labels = "High School", "Graduate","Uneducated","Unknown","College","Post-Graduate","Doctorate"
    #         size = 0.5
    #         fig3,axx1=plt.subplots(figsize=(12, 12))
    #         wedges, texts, autotexts = plt.pie([data["Education_Level"].value_counts()[0],
    #                                             data["Education_Level"].value_counts()[1],
    #                                             data["Education_Level"].value_counts()[2],
    #                                             data["Education_Level"].value_counts()[3],
    #                                             data["Education_Level"].value_counts()[4],
    #                                             data["Education_Level"].value_counts()[4],
    #                                             data["Education_Level"].value_counts()[6],

    #                                             ],

    #                                         explode=(0, 0,0,0,0,0,0),
    #                                         textprops=dict(size=20, color="white"),
    #                                         autopct="%.2f%%",
    #                                         pctdistance=0.72,
    #                                         radius=.9,
    #                                         colors=["#682F2F", "#F3AB60", "#9F8A78","#EED5B7","#00008B","#EE7621","#006400"],
    #                                         shadow=True,
    #                                         wedgeprops=dict(width=size, edgecolor="black",
    #                                                         linewidth=4),
    #                                         startangle=20)

    #         plt.legend(wedges, labels, title="Trình độ học vấn", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
    #                 edgecolor="black")
    #         st.pyplot(fig3)
            
    #         labels = "Married", "Unknown","Single","Divorced"
    #         size = 0.5
    #         fig4,axx2=plt.subplots(figsize=(12, 12))
    #         wedges, texts, autotexts = plt.pie([data["Marital_Status"].value_counts()[0],
    #                                             data["Marital_Status"].value_counts()[1],
    #                                             data["Marital_Status"].value_counts()[2],
    #                                             data["Marital_Status"].value_counts()[3],

    #                                             ],

    #                                         explode=(0, 0,0,0),
    #                                         textprops=dict(size=20, color="white"),
    #                                         autopct="%.2f%%",
    #                                         pctdistance=0.72,
    #                                         radius=.9,
    #                                         colors=["#682F2F", "#F3AB60", "#9F8A78","#EED5B7"],
    #                                         shadow=True,
    #                                         wedgeprops=dict(width=size, edgecolor="black",
    #                                                         linewidth=4),
    #                                         startangle=20)

    #         plt.legend(wedges, labels, title="Tình trạng hôn nhân", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
    #                 edgecolor="black")
    #         st.pyplot(fig4)
    #         ######
            
    #         st.write('Phân bố độ tuổi khách hàng')
    #         fig11,ax=plt.subplots(figsize=(20, 8))
    #         p = sns.histplot(data["Customer_Age"],color="#682F2F")
    #         plt.ylabel("Count",fontsize=20)
    #         plt.xlabel("\nAge",fontsize=20)
    #         sns.despine(left=True, bottom=True)
    #         st.pyplot(fig11)
    #         st.write('Phân bố độ tuổi khách hàng')
if selected=='Phân cụm':
    
            
    tab1, tab2 = st.tabs(["Phân cụm", "Biểu đồ"])
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    with tab1:
        # Lets drop the ID column which dosent provide any info but a sequentail order
        # ['PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','PURCHASES_TRX']
        # data.drop(columns= 'CUST_ID', axis = 1, inplace= True)
        # data.drop(columns= 'MINIMUM_PAYMENTS', axis = 1, inplace= True)
        X = data[['PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','PURCHASES_TRX']]
#['BALANCE','PURCHASES','PURCHASES_FREQUENCY','PAYMENTS']
        cols = X.columns#
            
            ####

    
        
# with tab2:    
        
        # Setup the color themes to differentiate clusters
        palette = ["#00aedb", "#a200ff", "#f47835", "#d41243", "#8ec127"]
        sns.set_palette(palette)
        cmap = ListedColormap(["#00aedb", "#a200ff", "#f47835", "#d41243", "#8ec127"])
        scaler = StandardScaler()
        scaler.fit(X)
        scaled_data = pd.DataFrame(scaler.transform(X),columns=X.columns)
        # scaled_data.head().T
        st.subheader('Tìm số thành phần chính')

        X = StandardScaler().fit_transform(X)

        X_mean = np.mean(X, axis=0)
        cov_mat = (X - X_mean).T.dot((X - X_mean)) / (X.shape[0]-1)
        # st.write('Ma trận hiệp phương sai')
        #st.dataframe(cov_mat)

        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        # st.dataframe( eig_vals)
        #st.write("Vector riêng")

        #st.dataframe(eig_vecs)
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low


        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        k=0
        num_=[]
        
        for i in eig_pairs:
            k=k+1
            
            num_.append(k)

        
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        
        cum_var_exp=pd.DataFrame(
        {    'n_component':num_,
            "Explained variance":cum_var_exp})
        st.write('Explained variance')
        
        st.dataframe(cum_var_exp)
        so_thanh_phan_chinh=st.sidebar.number_input('Lựa chọn số thành phần chính',value= 3, min_value=2, max_value=len(X[0]))
        
    # Linear dimensionality reduction using Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space with 3 features.
        pca = PCA(n_components=so_thanh_phan_chinh)
        column=["feature_"+str(i) for i in range(1, so_thanh_phan_chinh+1)]
        pca.fit(scaled_data)
        pca_data = pd.DataFrame(pca.transform(scaled_data), columns=column)
        #pca_data.describe().T
        wcss = []

    
        elbow=st.sidebar.button("Elbow")
        so_cum=st.sidebar.number_input('Lựa chọn số cụm',value= 3, min_value=2)

        if elbow:
            fige, axes = plt.subplots(figsize=(20, 8))

            elbow = KElbowVisualizer(KMeans(), k=10, timings=False, locate_elbow=True, size=(1260, 450))
            data1 = StandardScaler().fit_transform(data[cols])

            elbow.fit(pca_data)

            axes.set_xlabel("\nK", fontsize=20)
            axes.set_ylabel("\nDistortion Score", fontsize=20)

            sns.despine(left=True, bottom=True)
            st.pyplot(fige)
            
            
            figgg = plt.figure(figsize=(20,20))
            ax1 = plt.subplot(221, projection='3d')
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)
            
        # Fit the K-Means Clustering model with 5 clusters
        kmeans = KMeans(n_clusters=so_cum, random_state=0).fit(X)
        # Add the cluster labels to the dataframes
        pca_data["clustering"] = kmeans.labels_
        scaled_data["clustering"]= kmeans.labels_
        data["clustering"]= kmeans.labels_
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)
            
        with open('scaled_data.pkl', 'wb') as f1:
            pickle.dump(scaled_data, f1)
            
        with open('pca_data.pkl', 'wb') as f2:
            pickle.dump(X, f2)
            
            
        # updatedata=st.sidebar.button("Cập nhật dữ liệu")
        # if updatedata:
        #     connection = pymysql.connect(
        #     host="127.0.0.1",
        #     user="root",
        #     password="123456",
        #     database="dss")
        #     with open('data.pkl', 'rb') as f:
        #         data=pickle.load(f)
        #         if data is not None:
    
        #             cursor = connection.cursor()

        #             insert_query = "DELETE FROM  dss.date_time"
        #             cursor.execute(insert_query)
        #             # Đẩy giữ liệu vào dim_date
        #             now = datetime.now()
                    
        #             YEAR        = datetime.now().year     # the current year
        #             MONTH       = datetime.now().month    # the current month
        #             DATE        = datetime.now().day      # the current day
        #             HOUR        = datetime.now().hour   # the current hour
        #             MINUTE      = datetime.now().minute # the current minute
        #             SECONDS     = datetime.now().second #the current second
        #             date_id = DATE+MONTH+YEAR+HOUR+MINUTE+SECONDS
        #             #date_id = f'{YEAR}{MONTH}{DATE}{HOUR}{MINUTE}'

        #             cursor = connection.cursor()
        #             insert_query = "INSERT INTO dss.date_time(date_id,date_time, day, month,year) VALUES (%s, %s,%s,%s,%s)"
        #             cursor.execute(insert_query, (date_id,now, DATE, MONTH, YEAR))
        #             connection.commit()
        #             cursor.close()

        #             # Đẩy giữ liệu vào data
        #             cursor = connection.cursor()
        #             dataframe=data;
        #             dataframe = dataframe[dataframe.columns[:-1]]
        #             insert_query = "DELETE FROM  dss.customer"
        #             cursor.execute(insert_query)
                
        #             #dataframe=dataframe.drop(columns=['clutering'])
        #             with connection.cursor() as cursor:
        #                 for _, row in dataframe.iterrows():
        #                     insert_query = "INSERT INTO dss.customer(customer_id,Customer_Age,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category,Months_on_book,Contacts_Count_12_mon,Total_Trans_Ct) VALUES (%s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s)"
        #                     cursor.execute(insert_query,( tuple(row)))
        #             connection.commit()
        #             cursor.close()


        #             cluster  = conn.query(
        #                     'select* from dss.clustering')
        #             cluster['date_id']=date_id
        #             date_id=cluster['date_id']

        #             # Đẩy giữ liệu vào clustering
        #             cursor = connection.cursor()
        #             dataframe=data;
        #             insert_query = "DELETE FROM  dss.clustering"
        #             cursor.execute(insert_query)
        #             df= pd.DataFrame()
        #             clutering1 = dataframe[dataframe.columns[11]]
                    
        #             customer_id1 = dataframe[dataframe.columns[0]]
        #             df['customer_id']=customer_id1
        #             df['date_id']=date_id
        #             df['clutering']=clutering1

        #             with connection.cursor() as cursor:
        #                 for _, row in df.iterrows():
        #                     insert_query= "INSERT INTO dss.clustering(customer_id,date_id, clustering) VALUES (%s, %s,%s)"
        #                     cursor.execute(insert_query,( tuple(row)))
        #             connection.commit()
        #             cursor.close()
        # # # Đóng kết nối
        #             connection.close()

        #             st.success("Dữ liệu đã được cập nhật.")


            

    with tab2:

        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('scaled_data.pkl', 'rb') as f1:
            scaled_data = pickle.load(f1)
        with open('pca_data.pkl', 'rb') as f2:
            pca_data = pickle.load(f2)
        fixxx, axs1 = plt.subplots(ncols=so_cum,nrows=1, figsize=(20,80))
        for i in range(so_cum):
            data[data['clustering'] == i][cols].mean().plot.barh(ax=axs1[i], xlim=(-1.5, 3), figsize=(20,20), sharey=True, title='Group '+str(i+1))
        st.pyplot(fixxx)
    # Visualize the clusters in 4 subplots
        # Visualize the clusters in 4 subplots.
        # if scaled_data is not None and pca_data is not None
        if scaled_data is not None and pca_data is not None:
            st.subheader('Số lượng khách hàng từng cụm')
            palette = ["#682F2F", "#B9C2C9", "#EEE8CD", "#F2AB00","#98F5FF"],

            fig3,ax1= plt.subplots(figsize=(20, 8))
            p = sns.countplot(x=data["clustering"], palette=["#682F2F", "#B9C2C9", "#EEE8CD", "#698B69","#98F5FF"], saturation=1,
                                edgecolor="#1c1c1c", linewidth=2)
            p.axes.set_yscale("linear")
            p.axes.set_ylabel("Số lượng", fontsize=40)
            p.axes.set_xlabel("\nCluster", fontsize=40)
            p.axes.set_xticklabels(p.get_xticklabels(), rotation=0)
            for container in p.containers:
                p.bar_label(container, label_type="center", padding=6, size=30, color="black", rotation=0,
                            bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "orange", "edgecolor": "white", "linewidth": 4,
                                    "alpha": 1})

            sns.despine(left=True, bottom=True)
            st.pyplot(fig3)
            
            
            
            st.subheader('Đặc điểm tổng số tiền khách hàng đã chi tiêu cho các giao dịch mua hàng')
            
            
            figg,ax1= plt.subplots(figsize=(20, 8))
            sns.boxenplot(x=data["clustering"], y=data["PURCHASES"], palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])

            ax1.set_ylabel("PURCHASES", fontsize=40)
            ax1.set_xlabel("\nCluster", fontsize=40)

            sns.despine(left=True, bottom=True)
            st.pyplot(figg)
            
            
            
            st.subheader('Đặc điểm tần suất nạp tiền vào thẻ của khách hàng')
            
            
            figg,ax1= plt.subplots(figsize=(20, 8))
            sns.boxenplot(data=data,x=data["clustering"], y=data["BALANCE_FREQUENCY"], palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])

            ax1.set_ylabel("BALANCE_FREQUENCY", fontsize=40)
            ax1.set_xlabel("\nCluster", fontsize=40)

            sns.despine(left=True, bottom=True)
            st.pyplot(figg)
            
            # vẽ 2 biểu đồ so sánh 2 variable với nhau 
            #sns.countplot(data=df, x="class", hue="alive")
            #sns.barplot(x=data["clustering"], y=data['ONEOFF_PURCHASES_FREQUENCY'] + data['PURCHASES_INSTALLMENTS_FREQUENCY'],palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])

            st.subheader('Đặc điểm tần suất mua hàng trả toàn bộ của khách hàng')
            
            
            figg,ax1= plt.subplots(figsize=(20, 8))
            sns.boxenplot(x=data["clustering"], y=data['ONEOFF_PURCHASES_FREQUENCY'] ,palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])
            #,hue= data['PURCHASES_INSTALLMENTS_FREQUENCY']
            ax1.set_ylabel("FREQUENCY", fontsize=40)
            ax1.set_xlabel("\nCluster", fontsize=40)

            sns.despine(left=True, bottom=True)
            st.pyplot(figg)
            st.subheader('Đặc điểm tần suất mua hàng trả góp của khách hàng')
            
            
            figg,ax1= plt.subplots(figsize=(20, 8))
            sns.boxenplot(x=data["clustering"], y=data['PURCHASES_FREQUENCY'] ,palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])
            #,hue= data['PURCHASES_INSTALLMENTS_FREQUENCY']
            ax1.set_ylabel("PURCHASES FREQUENCY", fontsize=40)
            ax1.set_xlabel("\nCluster", fontsize=40)

            sns.despine(left=True, bottom=True)
            st.pyplot(figg)
            #TENURE
            
            # st.subheader('Đặc điểm thời gian sử dụng thẻ')
            
            
            # figg,ax1= plt.subplots(figsize=(20, 8))
            # sns.boxenplot(x=data["clustering"], y=data['TENURE'] ,palette=["#B9C0C9", "#682F2F", "#B9C2C9", "#F3AB60","#EEE8CD"])
            # #,hue= data['PURCHASES_INSTALLMENTS_FREQUENCY']
            # ax1.set_ylabel("Thời gian sử dụng", fontsize=40)
            # ax1.set_xlabel("\nCluster", fontsize=40)

            # sns.despine(left=True, bottom=True)
            # st.pyplot(figg)
            