import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Streamlit Classification App")
#/Users/skylarshi/Documents/Streamlit/dataset.csv
# Sidebar options
st.sidebar.header("Options:")
csv_file_path = st.sidebar.text_input("Enter CSV File Path:")
if csv_file_path:
    data = pd.read_csv(csv_file_path)
    st.write(data.head())
    st.session_state.data = data  # Store data in session state

st.sidebar.text_input("Enter Target Variable:", key='target_variable')

# Buttons
button1 = st.sidebar.button("Split Dataset X and Y")
button2 = st.sidebar.button("Split Dataset into Training and Testing")
model_selection = st.sidebar.selectbox("Select Model:", ["KNN Classifier", "Decision Tree Classifier"], key='model_selection')
button3 = st.sidebar.button("Run Model and Analyze Results")

# Split Dataset X and Y
if button1 and 'data' in st.session_state and st.session_state.target_variable:
    X = st.session_state.data.drop(st.session_state.target_variable, axis=1)
    Y = st.session_state.data[st.session_state.target_variable]
    st.write(X.head())
    st.write(Y.head())
    st.session_state.X = X
    st.session_state.Y = Y

# Split Dataset into Training and Testing
if button2 and 'X' in st.session_state and 'Y' in st.session_state:
    X_train, X_test, Y_train, Y_test = train_test_split(st.session_state.X, st.session_state.Y, test_size=0.2, random_state=42)
    st.write(X_train.head())
    st.write(X_test.head())
    st.write(Y_train.head())
    st.write(Y_test.head())
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.Y_train = Y_train
    st.session_state.Y_test = Y_test

# Run Model and Analyze Results
if button3 and 'X_train' in st.session_state and 'Y_train' in st.session_state:
    if st.session_state.model_selection == "KNN Classifier":
        model = KNeighborsClassifier()
    elif st.session_state.model_selection == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    model.fit(st.session_state.X_train, st.session_state.Y_train)
    Y_pred = model.predict(st.session_state.X_test)
    st.write(Y_pred)
    # Report the accuracy
    st.write(accuracy_score(st.session_state.Y_test, Y_pred))
    # Report the confusion matrix
    st.write(confusion_matrix(st.session_state.Y_test, Y_pred))