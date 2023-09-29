import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import base64

st.title("""Оценка качества фичей""")
uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Загружен следующий DataFrame:", df)

    selected_features = st.sidebar.multiselect("Выберите столбцы как признаки (features)", df.columns)

    # Предлагаем пользователю выбрать целевую переменную (target)
    target_column = st.sidebar.selectbox("Выберите столбец как целевую переменную (target)", df.columns)
    if selected_features is not None and target_column is not None:
        selected_epochs = st.sidebar.number_input("Выберите число эпох :", min_value=1, max_value=10000, value=100)
        selected_learningrate = st.sidebar.slider("Выберите шаг обучения:", min_value=0.00, max_value=1.0, value=0.1, step=0.05)
        x_train = df[selected_features]
        y_train = df[target_column]
        if x_train is not None and not x_train.empty:
            ss = StandardScaler()
            x_train = ss.fit_transform(x_train)
            def sigmoid(x):
                return 1/(1+np.exp(-x))

            class LinReg:
                def __init__(self, learning_rate, n_inputs):
                    self.learning_rate = learning_rate
                    self.n_inputs = n_inputs
                    self.coef_ = np.random.rand(n_inputs)
                    self.intercept_ = np.random.rand()
                
                def fit(self, X, y, n_iterations):
                    for _ in range(n_iterations):
                        predictions = sigmoid(np.dot(X, self.coef_) + self.intercept_)
                        gradient_coef = (1/ len(X)) * np.dot(X.T , (predictions - y))
                        gradient_intercept = (1/ len(X)) * np.sum(predictions - y)
                        self.coef_ -= self.learning_rate * gradient_coef
                        self.intercept_ -= self.learning_rate * gradient_intercept

                def predict(self, X):
                    return sigmoid(np.dot(X, self.coef_) + self.intercept_)
                    
                def class_pred(self, X,y):
                    y_pred = self.predict(X)
                    class_pred = [0 if y<=0.5 else 1 for y in y_pred]
                    return class_pred

                def score(self, X, y):
                    y_pred = self.predict(X)
                    return -(((y*np.log(y_pred))+((1-y)*np.log(1-y_pred))).mean())
                
            test_obj = LinReg(selected_learningrate, x_train.shape[1])
            test_obj.fit(x_train,y_train,selected_epochs) 
            #my_dict = {"x1": test_obj.coef_[0], "x2": test_obj.coef_[1], "x3": test_obj.coef_[2], "Свободный член": test_obj.intercept_}
            my_dict = {}
            for i, coef in enumerate(test_obj.coef_):
                key = f"x{i + 1}" 
                my_dict[key] = coef

            my_dict["Свободный член"] = test_obj.intercept_

            st.write("Оценка наших фичей:", my_dict)

            pred_csv = test_obj.class_pred(x_train,y_train)
            df_pred = pd.DataFrame(pred_csv)
            df_pred.rename(columns={0: 'Предсказание'}, inplace=True)
            st.write("Можете скачать файл с предсказаниями для ваших фичей:")
            if st.button("Скачать CSV"):
                csv = df_pred.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Скачать CSV файл</a>'
                st.markdown(href, unsafe_allow_html=True)


            st.write("Пора строить графики, давай-ка выберем фичи, которые хочешь глянуть")


            selected_features_for_graphics = st.sidebar.multiselect("Выберите фичи для которых хочешь построить график 0 = x1и тд, я криворучка", range(x_train.shape[1]))
            if selected_features_for_graphics:
                fig1 = plt.figure(figsize=(10,6))
                plt.scatter(x_train[:,selected_features_for_graphics[0]], x_train[:,selected_features_for_graphics[1]])
                plt.title('Зависимость фичей')
                plt.xlabel(f'Фича {selected_features_for_graphics[0]}')
                plt.ylabel(f'Фича {selected_features_for_graphics[1]}')
                st.pyplot(fig1)

                fig2 = plt.figure(figsize=(10,6))
                plt.plot(x_train[:,selected_features_for_graphics[0]], x_train[:,selected_features_for_graphics[1]])
                plt.title('Зависимость фичей')
                plt.xlabel(f'Фича {selected_features_for_graphics[0]}')
                plt.ylabel(f'Фича {selected_features_for_graphics[1]}')
                st.pyplot(fig2)

                fig3 = plt.figure(figsize=(10,6))
                plt.bar(x_train[:,selected_features_for_graphics[0]], x_train[:,selected_features_for_graphics[1]])
                plt.title('Зависимость фичей')
                plt.xlabel(f'Фича {selected_features_for_graphics[0]}')
                plt.ylabel(f'Фича {selected_features_for_graphics[1]}')
                st.pyplot(fig3)
                