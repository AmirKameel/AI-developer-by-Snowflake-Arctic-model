import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
from dotenv import load_dotenv
from pandasai import Agent
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
from sklearn import metrics
import io
import joblib
from sklearn import metrics
from scipy.stats import skew, kurtosis
from scipy.stats import pearsonr, spearmanr
import streamlit as st
import replicate
import os


# Define StreamlitCallback class
class StreamlitCallback:
    def __init__(self, container):
        self.container = container

    def show_code(self, code):
        self.container.code(code)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "data_analysis"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Title of the app
st.title("AI Fullstack Developer 🦹‍♂")

# Navigation sidebar
page = st.sidebar.radio(
    "Choose a page:", 
    ["**Data Analysis 📊**", "**Data Visualization 📈**", "**Machine Learning 🤖**", "**Blow Your Mind With Arctic 🧑⚡**"], 
    index=0
)

# Choose problem type
problem_type = st.sidebar.selectbox("Select Problem Type", ["Regression", "Classification"])

if page == "**Data Analysis 📊**":
    st.session_state.page = "data_analysis"
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Display dataset information
    if st.session_state.df is not None:
        st.subheader("Dataset Information")
        st.write(st.session_state.df.describe())

        # Select features and target column
        feature_columns = st.multiselect("Select Feature Columns", st.session_state.df.columns.tolist())
        target_column = st.selectbox("Select Target Column", st.session_state.df.columns.tolist())

        # Store selected feature columns and target column in session state
        st.session_state.feature_columns = feature_columns
        st.session_state.target_column = target_column

        # Reorder columns with target column last
        columns_to_display = feature_columns + [target_column]
        if st.session_state.df is not None:
            st.session_state.df = st.session_state.df[columns_to_display]

        st.dataframe(st.session_state.df.head())

        # Data Preprocessing
        st.subheader("Data Preprocessing")

        # Separate columns by data type
        text_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Display report before preprocessing
        if st.button("Generate Preprocessing Report"):
            st.session_state.report_generated = True
            st.session_state.report_data = []

            for col in st.session_state.df.columns:
                data_type = st.session_state.df[col].dtype
                null_count = st.session_state.df[col].isnull().sum()
                unique_values = st.session_state.df[col].nunique()
                duplicate_count = st.session_state.df.duplicated(subset=[col]).sum()

                if data_type in ['float64', 'int64']:
                    outliers = len(st.session_state.df[(np.abs(st.session_state.df[col] - st.session_state.df[col].mean()) > 3 * st.session_state.df[col].std())])
                else:
                    outliers = ''

                # Append dictionary to the list
                st.session_state.report_data.append({'Column': col, 'Data Type': data_type, 'Null Values': null_count,
                                                     'Unique Values': unique_values, 'Duplicates': duplicate_count,
                                                     'Outliers': outliers})

            # Convert list of dictionaries into DataFrame
            report_df = pd.DataFrame(st.session_state.report_data)

            st.write(report_df)

            # Download report as CSV
            csv_file = report_df.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessing_report.csv">Download Preprocessing Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        # If report generated, display preprocessing options
        if getattr(st.session_state, 'report_generated', False):
            # Text Columns Preprocessing
            if text_columns:
                st.subheader("Text Columns Preprocessing")
                selected_text_columns = st.multiselect("Select Text Columns", text_columns)

                if st.button("Remove Null Values"):
                    for column in selected_text_columns:
                        st.session_state.df[column].dropna(inplace=True)
                    st.write(f"Removed null values for selected text columns. Data shape: {st.session_state.df.shape}")
                    st.dataframe(st.session_state.df.head())

                if st.button("Remove Duplicates"):
                    for column in selected_text_columns:
                        st.session_state.df.drop_duplicates(subset=[column], inplace=True)
                    st.write(f"Removed duplicates for selected text columns. Data shape: {st.session_state.df.shape}")
                    st.dataframe(st.session_state.df.head())

                encoding_option = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
                if encoding_option == "Label Encoding":
                    for column in selected_text_columns:
                        le = LabelEncoder()
                        st.session_state.df[column] = le.fit_transform(st.session_state.df[column])
                    st.write("Applied Label Encoding on selected text columns.")
                    st.dataframe(st.session_state.df.head())
                elif encoding_option == "One-Hot Encoding":
                    for column in selected_text_columns:
                        encoder = OneHotEncoder(sparse=False, drop="first")
                        encoded_cols = pd.DataFrame(encoder.fit_transform(st.session_state.df[[column]]),
                                                     columns=encoder.get_feature_names_out([column]))
                        st.session_state.df = pd.concat([st.session_state.df, encoded_cols], axis=1)
                        st.session_state.df.drop(columns=[column], inplace=True)
                    st.write("Applied One-Hot Encoding on selected text columns.")
                    st.dataframe(st.session_state.df.head())

            # Numerical Columns Preprocessing
            if numerical_columns:
                st.subheader("Numerical Columns Preprocessing")
                selected_numerical_columns = st.multiselect("Select Numerical Columns", numerical_columns)

                # Additional preprocessing options
                if st.checkbox("Handle Missing Values"):
                    method = st.selectbox("Select Method", ["Mean", "Median", "Drop"])
                    if method == "Mean":
                        st.session_state.df[selected_numerical_columns] = st.session_state.df[selected_numerical_columns].fillna(st.session_state.df[selected_numerical_columns].mean())
                    elif method == "Median":
                        st.session_state.df[selected_numerical_columns] = st.session_state.df[selected_numerical_columns].fillna(st.session_state.df[selected_numerical_columns].median())
                    else:
                        st.session_state.df = st.session_state.df.dropna(subset=selected_numerical_columns)
                    st.write("Handled missing values for selected numerical columns.")
                    st.dataframe(st.session_state.df.head())

                if st.checkbox("Handle Outliers"):
                    method = st.selectbox("Select Method", ["Remove", "Clip"])
                    if method == "Remove":
                        for column in selected_numerical_columns:
                            st.session_state.df = st.session_state.df[(np.abs(st.session_state.df[column] - st.session_state.df[column].mean()) <= 3 * st.session_state.df[column].std())]
                    else:
                        for column in selected_numerical_columns:
                            lower_bound = st.session_state.df[column].mean() - (3 * st.session_state.df[column].std())
                            upper_bound = st.session_state.df[column].mean() + (3 * st.session_state.df[column].std())
                            st.session_state.df[column] = np.clip(st.session_state.df[column], lower_bound, upper_bound)
                    st.write("Handled outliers for selected numerical columns.")
                    st.dataframe(st.session_state.df.head())

                if st.checkbox("Scaling"):
                    scaling_method = st.selectbox("Select Scaling Method", ["Standardization", "Normalization"])
                    if scaling_method == "Standardization":
                        scaler = StandardScaler()
                        st.session_state.df[selected_numerical_columns] = scaler.fit_transform(st.session_state.df[selected_numerical_columns])
                        st.write("Applied Standardization on selected numerical columns.")
                    else:
                        st.session_state.df[selected_numerical_columns] = (st.session_state.df[selected_numerical_columns] - st.session_state.df[selected_numerical_columns].min()) / (st.session_state.df[selected_numerical_columns].max() - st.session_state.df[selected_numerical_columns].min())
                        st.write("Applied Normalization on selected numerical columns.")
                    st.dataframe(st.session_state.df.head())

        # Allow users to convert datatype of any column
        st.subheader("Convert Datatype of Columns")
        column_to_convert = st.selectbox("Select Column to Convert", st.session_state.df.columns.tolist())
        new_datatype = st.selectbox("Select New Datatype", ["int", "float", "object", "datetime", "category"])

        if st.button("Convert Datatype"):
            try:
                if new_datatype == "int":
                    st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(int)
                elif new_datatype == "float":
                    st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(float)
                elif new_datatype == "object":
                    st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(str)
                elif new_datatype == "datetime":
                    st.session_state.df[column_to_convert] = pd.to_datetime(st.session_state.df[column_to_convert])
                elif new_datatype == "category":
                    st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype("category")
                
                st.success(f"Successfully converted datatype of column '{column_to_convert}' to {new_datatype}.")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")


elif page == "**Data Visualization 📈**":
    st.session_state.page = "data_visualization"

    st.subheader("Data Visualization")

    if st.session_state.df is not None:  # Check if df is loaded
        # Correlation Heatmap
        st.write("**Correlation Heatmap**")
        corr = st.session_state.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(plt)

        # Pairplot for selected features
        if st.session_state.feature_columns:
            if len(st.session_state.feature_columns) > 1:
                st.write("**Pairplot for Selected Features**")
                pairplot = sns.pairplot(st.session_state.df[st.session_state.feature_columns])
                st.pyplot(pairplot)

        # Histograms for numerical columns
        if st.button("Generate Histograms for Numerical Columns"):
            numerical_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for column in numerical_columns:
                plt.figure(figsize=(8, 6))
                st.session_state.df[column].hist(bins=20)
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {column}")
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)

        # Scatter plots for numerical columns
        if st.button("Generate Scatter Plots for Numerical Columns"):
            numerical_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for col1 in numerical_columns:
                for col2 in numerical_columns:
                    if col1 != col2:
                        plt.figure(figsize=(8, 6))
                        sns.regplot(data=st.session_state.df, x=col1, y=col2)
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.title(f"Regression Plot between {col1} and {col2}")
                        st.pyplot()
                        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Boxplots for numerical columns
        if st.button("Generate Boxplots for Numerical Columns"):
            numerical_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for column in numerical_columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=st.session_state.df, x=column)
                plt.xlabel(column)
                plt.title(f"Boxplot of {column}")
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)

    else:
        st.write("Please upload a dataset in the Data Analysis page.")



elif page == "**Machine Learning 🤖**":
    st.session_state.page = "machine_learning"

    st.subheader("Machine Learning")

    if st.session_state.df is not None:  
        # Choose machine learning model
        if problem_type == "Regression":
            st.write("**Regression Models**")
            model_option = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regression", "Gradient Boosting Regression"])

            if model_option == "Linear Regression":
                st.write("**Linear Regression**")
                model = LinearRegression()
            elif model_option == "Random Forest Regression":
                st.write("**Random Forest Regression**")
                model = RandomForestRegressor()
            elif model_option == "Gradient Boosting Regression":
                st.write("**Gradient Boosting Regression**")
                model = GradientBoostingRegressor()

        elif problem_type == "Classification":
            st.write("**Classification Models**")
            model_option = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classification", "Gradient Boosting Classification"])

            if model_option == "Logistic Regression":
                st.write("**Logistic Regression**")
                model = LogisticRegression()
            elif model_option == "Random Forest Classification":
                st.write("**Random Forest Classification**")
                model = RandomForestClassifier()
            elif model_option == "Gradient Boosting Classification":
                st.write("**Gradient Boosting Classification**")
                model = GradientBoostingClassifier()

        # Train-test split
        X = st.session_state.df[st.session_state.feature_columns]
        y = st.session_state.df[st.session_state.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set model parameters
        st.subheader("Set Model Parameters")

        if problem_type == "Regression":
            if model_option == "Random Forest Regression":
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, step=10, value=100)
                max_depth = st.slider("Max Depth", min_value=1, max_value=20, step=1, value=10)

            elif model_option == "Gradient Boosting Regression":
                n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, step=50, value=100)
                learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.2, step=0.01, value=0.1)
                max_depth = st.slider("Max Depth", min_value=1, max_value=10, step=1, value=3)

        elif problem_type == "Classification":
            if model_option == "Random Forest Classification":
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, step=10, value=100)
                max_depth = st.slider("Max Depth", min_value=1, max_value=20, step=1, value=10)

            elif model_option == "Gradient Boosting Classification":
                n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, step=50, value=100)
                learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.2, step=0.01, value=0.1)
                max_depth = st.slider("Max Depth", min_value=1, max_value=10, step=1, value=3)

        # Fine-tune the model
        if st.button("Fine-Tune Model"):
            st.subheader("Fine-Tuning Model")

            if problem_type == "Regression":
                if model_option == "Random Forest Regression":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                elif model_option == "Gradient Boosting Regression":
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

            elif problem_type == "Classification":
                if model_option == "Random Forest Classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                elif model_option == "Gradient Boosting Classification":
                    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

            # Model training
            model.fit(X_train, y_train)

            # Model evaluation and visualization
            if problem_type == "Regression":
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate training and testing accuracies
                train_accuracy = model.score(X_train, y_train)
                test_accuracy = model.score(X_test, y_test)

                # Calculate mean squared error on the test set
                test_mse = mean_squared_error(y_test, y_pred_test)

                st.write("**Regression Metrics after Fine-Tuning**")
                metrics_df = pd.DataFrame({
                    "Metric": ["Mean Squared Error", "Mean Absolute Error", "R-squared Score"],
                    "Value": [test_mse, mean_absolute_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
                })
                st.table(metrics_df)

                # Plot training and testing accuracies
                #plt.figure(figsize=(10, 6))
                #plt.plot(X_train, y_train, label='Training Data', marker='o')
                #plt.plot(X_test, y_test, label='Testing Data', marker='o')
                #plt.xlabel('Input')
                #plt.ylabel('Output')
                #plt.title('Regression Model: Training vs Testing Data')
                #plt.legend()
                #plt.grid(True)
                #st.pyplot()

        
                
            elif problem_type == "Classification":
                y_pred = model.predict(X_test)
                st.write("**Classification Metrics after Fine-Tuning**")
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy Score"],
                    "Value": [accuracy_score(y_test, y_pred)]
                })
                st.table(metrics_df)

                # Visualize confusion matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot()
                #cm_display.plot()
                #plt.show()

            # Allow model download
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            st.download_button(
                label="Download Model",
                data=buffer,
                file_name=f"{model_option.replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream"
            )

    else:
        st.write("Please upload a dataset in the Data Analysis page.")





elif page == "**Blow Your Mind 🧑⚡**":
    st.session_state.page = "interact_with_data"

    if st.session_state.df is not None:
        st.subheader("Just think and ask then let Arctic do the rest 😉")
        
        selected_columns = st.multiselect("Select columns", st.session_state.df.columns)
        query = st.text_input("Write your query")
        replicate_api_key = st.text_input("Enter your Replicate API Key", type='password')

        if st.button("Submit"):
            if not selected_columns:
                st.error("Please select at least one column.")
            elif not query:
                st.error("Please write your query.")
            elif not replicate_api_key:
                st.error("Please enter your Replicate API Key.")
            else:
                # Your project code starts here
                import replicate
                import os
                from transformers import AutoTokenizer

                # Set assistant icon to Snowflake logo
                icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}

                # Replicate Credentials
                with st.sidebar:
                    st.title('Snowflake Arctic')
                    os.environ['REPLICATE_API_TOKEN'] = replicate_api_key
                    st.subheader("Adjust model parameters")
                    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
                    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

                # Function to generate Snowflake Arctic response
                def generate_arctic_response(prompt):
                    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
                    tokens = tokenizer.tokenize(prompt)
                    if len(tokens) >= 2048:
                        return "Conversation length too long. Please keep it under 2048 tokens."
                    else:
                        for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                                      input={"prompt": prompt,
                                                             "prompt_template": r"{prompt}",
                                                             "temperature": temperature,
                                                             "top_p": top_p}):
                            yield str(event)

                # Prepare prompt for Snowflake Arctic
                prompt = f"user\n"
                for column in selected_columns:
                    prompt += f"{column}: {', '.join(map(str, st.session_state.df[column].tolist()))}\n"
                prompt += f"{query}\nassistant\n"

                # Generate response from Snowflake Arctic
                response = generate_arctic_response(prompt)

                # Display Snowflake Arctic response
                st.write("### Model Response")
                st.write_stream(response)
                # Your project code ends here

    else:
        st.write("Please upload a dataset in the Data Analysis page.")
