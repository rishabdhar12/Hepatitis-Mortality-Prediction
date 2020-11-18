import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os
plt.style.use('fivethirtyeight')

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model

feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 
'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 
'sgot', 'albumin', 'protime','histology']

gender_dict = {"male":1,"female":2}
feature_dict = {"No":1,"Yes":2}
f_dict = {1:"No",2:"Yes"}

def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key


def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 


def main():
    st.title('PATIENT MORTALITY PREDICTOR')
    st.subheader("CLASSIFY WHETHER THE PATIENT'S GONNA LIVE OR DIE")

    activities = ['EXPLORATORY DATA ANALYSIS','MAKE PREDICTION']
    choices = st.sidebar.selectbox('ACTIVITIES',activities)

    if choices == 'EXPLORATORY DATA ANALYSIS':
        st.subheader('VISUALIZE THE DATA')
        df = pd.read_csv('Model/hepatitis.csv')
        st.dataframe(df)

        if st.button('DESCRIBE'):
            df.describe().T

        plots = ['UNIVARIATE ANALYSIS','BIVARIATE ANALYSIS']
        select = st.selectbox('PLOTS',plots)
        if select == 'UNIVARIATE ANALYSIS':
            if st.button('COUNTPLOTS'):
                cols = ['class','age', 'sex', 'steroid', 'antivirals','fatigue','spiders',
                'ascites', 'varices', 'bilirubin', 'albumin',
                'protime', 'histology']
                for col in cols:
                    #df[col].value_counts().plot(kind='bar')
                    plt.figure(figsize=(12,5))
                    sns.countplot(col, data=df)
                    plt.title('COUNT OF {}'.format(col.upper()), size=25)
                    plt.xticks(rotation=90)
                    plt.xlabel(xlabel=col.upper(), fontsize=17)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    #plt.tight_layout()
                    st.pyplot()

            if st.button('PIE PLOTS'):
                fig,ax = plt.subplots(1,2,figsize=(12,6))
                ax[0].pie(df['class'].value_counts(), labels=['DIE','LIVE'],explode=[0.01,0.01],
                colors=['yellow','orange'], autopct='%.2f%%',shadow=True)
                ax[1].pie(df['sex'].value_counts(), labels=['MALE','FEMALE'],explode=[0.01,0.01],
                colors=['red','blue'], autopct='%.2f%%',shadow=True)
                st.pyplot()

            if st.button('CORRELATION'):
                mask = np.zeros_like(df.corr())
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    fig, ax = plt.subplots(figsize=(15,9))
                    ax = sns.heatmap(df.corr(), mask=mask, vmax=.3, linewidths=.5)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()


        elif select == 'BIVARIATE ANALYSIS':
            if st.checkbox("Select Columns To Show"):
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect('Select',all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
            
            if st.checkbox('SCATTER PLOT'):
                # st.area_chart(new_df)
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect('Select',all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                for i in selected_columns:
                    for j in selected_columns:
                            if i != j:
                                sns.scatterplot(x=new_df[i], y=df[j])
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()
            
            if st.checkbox('BOXPLOT'):
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect('Select',all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                for i in selected_columns:
                    for j in selected_columns:
                            if i != j:
                                sns.boxplot(x=new_df[i], y=df[j])
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()

            

    elif choices == 'MAKE PREDICTION':
        st.subheader('predict')

        age = st.slider('Age',7,80)
        sex = st.radio('Sex',tuple(gender_dict.keys()))
        steroid = st.radio('Steriod', tuple(feature_dict.keys()))
        antivirals = st.radio('Antivirals', tuple(feature_dict.keys()))
        fatigue = st.radio('Fatigue', tuple(feature_dict.keys()))
        spiders = st.radio('Spiders', tuple(feature_dict.keys()))
        ascites = st.radio('Ascites', tuple(feature_dict.keys()))
        varices = st.radio('Varices', tuple(feature_dict.keys()))
        bilirubin = st.slider('Bilirubin',0.0,8.0)
        alk_phosphate = st.slider('Alk Phosphate',33,250)
        sgot = st.slider('SGOT',13,500)
        albumin = st.slider('Albumin',2.0,6.0)
        protime = st.slider('Protime',10,90)
        histology = st.radio('Histology', tuple(feature_dict.keys()))
        feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]
        #st.write(len(feature_list))
        #st.write(feature_list)
        pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
        st.json(pretty_result)
        single_sample = np.array(feature_list).reshape(1,-1)
        
        if st.button('PREDICT'):
            loaded_model = load_model('/Model/model.pkl')
            prediction = loaded_model.predict(single_sample)
            probability = loaded_model.predict_proba(single_sample)

            if prediction==2:
                st.success('PATIENT LIVES')
            else:
                st.warning('PATIENT DIES')









if __name__ == '__main__':
    main()
