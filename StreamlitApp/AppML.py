import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Set page configuration
st.set_page_config(page_title="Charity Donors Prediction", page_icon="💰", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2E86C1;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("# Charity Donors Prediction")
st.markdown("Web-Based Interface for Income Prediction and Data Analysis")

# Insert containers separated into tabs
tab1, tab2 = st.tabs(["Prediction", "Analysis"])

# Tab 1: Prediction
with tab1:
    st.title("Income Prediction")
    st.markdown("Use the form below to predict whether an individual's income is **≤50K** or **>50K**.")

    # Streamlit form for user input
    with st.form(key='income_form'):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=18, max_value=100, value=39)  # Rounded mean of 'age'
            education_level = st.selectbox('Education Level', ['Bachelors', 'Masters', 'PhD', 'High School', 'HS-grad'], index=4)  # Most frequent: 'HS-grad'
            education_num = st.number_input('Education Num', min_value=1, max_value=16, value=10)  # Rounded mean of 'education-num'
            workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'], index=0)  # Most frequent: 'Private'
            marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed'], index=1)  # Most frequent: 'Married-civ-spouse'
            occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'], index=1)  # Most frequent: 'Craft-repair'

        with col2:
            relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], index=2)  # Most frequent: 'Husband'
            race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], index=0)  # Most frequent: 'White'
            sex = st.selectbox('Sex', ['Female', 'Male'], index=1)  # Most frequent: 'Male'
            capital_gain = st.number_input('Capital Gain', min_value=0, max_value=1000000, value=0)  # Rounded mean of 'capital-gain'
            capital_loss = st.number_input('Capital Loss', min_value=0, max_value=1000000, value=0)  # Rounded mean of 'capital-loss'
            hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=41)  # Rounded mean of 'hours-per-week'
            native_country = st.selectbox('Native Country', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Puerto-Rico', 'Cuba', 'England', 'Jamaica'], index=0)  # Most frequent: 'United-States'

        # Submit button
        submit_button = st.form_submit_button(label='Predict Income')

        # Prepare input data when form is submitted
        if submit_button:
            new_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'education_level': [education_level],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })

            # Apply transformations to the input data
            new_data['capital-gain'] = np.log1p(new_data['capital-gain'])  # Apply log(x + 1) transformation
            new_data['capital-loss'] = np.log1p(new_data['capital-loss'])  # Apply log(x + 1) transformation

            # Scale numerical features
            scaler = MinMaxScaler()
            numerical_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num']
            new_data[numerical_columns] = scaler.fit_transform(new_data[numerical_columns])

            # Label encoding for binary columns
            le = LabelEncoder()
            new_data['sex'] = le.fit_transform(new_data['sex'])

            # One-hot encoding for multi-class categorical columns
            new_data = pd.get_dummies(new_data, columns=['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'],
                                    drop_first=True, dtype=int)

            # Load the feature names
            feature_names = joblib.load("feature_names.pkl")

            # Ensure the input data has the same columns as the training data
            new_data = new_data.reindex(columns=feature_names, fill_value=0)

            # Load the saved model
            model_pipeline = joblib.load("adaboost_model.pkl")

            # Make prediction using the loaded model pipeline
            prediction = model_pipeline.predict(new_data)

            # Display the prediction
            st.markdown("### Prediction Result")
            if prediction[0] == '<=50K':
                st.error('Income is **less than or equal to 50K**')
            else:
                st.success('Income is **greater than 50K**')

# Tab 2: Analysis
with tab2:
    st.title("Data Analysis")
    st.markdown("Explore the dataset and gain insights through interactive visualizations.")

    # Load the dataset
    data = pd.read_csv("census.csv")

    st.write("### Census Dataset Preview")
    st.dataframe(data.head())

    # Distribution of age
    st.write("### Age Distribution")
    fig1 = px.histogram(data, x='age', nbins=30, color_discrete_sequence=['#3498DB'], title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Distribution of hours-per-week
    st.write("### Hours-per-week Distribution")
    fig2 = px.histogram(data, x='hours-per-week', nbins=30, color_discrete_sequence=['#2ECC71'], title="Hours-per-week Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # Income vs. Age
    st.write("### Income vs. Age")
    fig3 = px.box(data, x='income', y='age', color='income', color_discrete_sequence=['#E74C3C', '#8E44AD'], title="Income vs. Age")
    st.plotly_chart(fig3, use_container_width=True)

    # Income Distribution by Education Level
    st.write("### Income Distribution by Education Level")
    fig4 = px.histogram(data, x='education_level', color='income', barmode='group', color_discrete_sequence=['#F1C40F', '#16A085'], title="Income Distribution by Education Level")
    st.plotly_chart(fig4, use_container_width=True)

    # Marital status vs. Income
    st.write("### Marital Status vs. Income")
    fig5 = px.histogram(data, x='marital-status', color='income', barmode='group', color_discrete_sequence=['#F39C12', '#27AE60'], title="Income Distribution by Marital Status")
    st.plotly_chart(fig5, use_container_width=True)

    # Occupation vs. Income
    st.write("### Occupation vs. Income")
    fig6 = px.histogram(data, x='occupation', color='income', barmode='group', color_discrete_sequence=['#D35400', '#2980B9'], title="Income Distribution by Occupation")
    st.plotly_chart(fig6, use_container_width=True)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr_matrix = data.select_dtypes(include=['number']).corr()
    fig7 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    ))
    fig7.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig7, use_container_width=True)

    st.write("### Insights:")
    st.markdown("""
    - **Age Distribution**: Most individuals are between 20 and 50 years old.
    - **Hours-per-week**: The majority work around 40 hours per week.
    - **Income vs. Age**: Higher income is more common among older individuals.
    - **Education Level**: Higher education levels correlate with higher income.
    - **Marital Status**: Married individuals tend to have higher incomes.
    - **Occupation**: Managerial and professional roles have higher incomes.
    - **Correlation Heatmap**: Highlights relationships between numerical features.
    """)