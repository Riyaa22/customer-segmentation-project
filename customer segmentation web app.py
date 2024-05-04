import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the trained model
pickle_in = open("cluster.pkl", "rb")
clf = pickle.load(pickle_in)

# Function to establish connection to MySQL database
def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            database='ABC',
            user='root',
            password='riya'
        )
        if conn.is_connected():
            return conn
        else:
            st.error("Failed to connect to database")
            return None
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

# Function to create 'feedback' table if it doesn't exist
def create_feedback_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                feedback_text TEXT
            )
        """)
        conn.commit()
        
    except Error as e:
        st.error(f"Error creating feedback table: {e}")

# Function to insert feedback into MySQL database
def insert_feedback(conn, feedback):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (feedback_text) VALUES (%s)", (feedback,))
        conn.commit()
        st.success("Thank you for your feedback!")
    except Error as e:
        st.error(f"Error inserting feedback into database: {e}")

def Customer_segmentation(Recency, Frequency, MonetaryValue):
    try:
        Recency = int(Recency)
        Frequency = int(Frequency)
        MonetaryValue = int(MonetaryValue)
        return clf.predict([[Recency, Frequency, MonetaryValue]])
    except ValueError:
        # Handle the case where the user enters a value that cannot be converted to an integer
        st.error("Please enter valid integer values for Recency, Frequency, and Monetary Value.")
        return None

# Function to display data visualization
def visualize_data(df):
    st.markdown("<h3 style='text-align: center;'>This can be Visualized as:</h3>",unsafe_allow_html=True)
    try:
        if not df.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].hist(df['Recency'], bins=20, color='blue', alpha=0.7)
            axes[0].set_title('Recency')

            axes[1].hist(df['Frequency'], bins=20, color='green', alpha=0.7)
            axes[1].set_title('Frequency')

            axes[2].hist(df['MonetaryValue'], bins=20, color='orange', alpha=0.7)
            axes[2].set_title('Monetary Value')

            st.pyplot(fig)
        else:
            st.write("No valid data to visualize.")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")

def main():
    try:
        st.set_page_config(
            page_title="CUSTOMER SEGMENTATION ANALYSIS",
            page_icon="📊",
        )
        
        st.title("Customer Segmentation Analysis📊")
        html_temp = """
        <div style="background-color:grey;padding:10px">
        <h2 style="color:black;text-align:center;">Machine Learning Web App</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        # Text inputs for user to enter data
        Recency = st.text_input("Enter Recency:","" )
        Frequency = st.text_input("Enter Frequency:","")
        MonetaryValue = st.text_input("Enter Monetary Value:","")

        # Button to predict customer segment
        if st.button("Predict the output"):
            if not Recency.isdigit() or not Frequency.isdigit() or not MonetaryValue.isdigit():
                st.error("Please enter valid integer values for Recency, Frequency, and Monetary Value.")
            else:
                result = Customer_segmentation(Recency, Frequency, MonetaryValue)
                if result is not None:
                    if result == 0:
                        result = "According to values, it is: New customer"
                    elif result == 1:
                        result = "According to values, Customer is at risk"
                    elif result == 2:
                        result = "According to values, it is: Best customer"
                    elif result == 3:
                        result = "According to values, Customer has been churned"
                    st.success(result)

        # Button to visualize input data
        if st.button("Visualize Data"):
           data = {
                'Recency': [int(Recency)],
                'Frequency': [int(Frequency)],
                'MonetaryValue': [int(MonetaryValue)]
            }
           df = pd.DataFrame(data)
           visualize_data(df)

        # Connect to the MySQL database
        conn = connect_to_database()
        if conn:
            # Create 'feedback' table if it doesn't exist
            create_feedback_table(conn)

            # Feedback section
            st.markdown("<h3 style='text-align: center;'>Feedback❤️</h3>",unsafe_allow_html=True)
            feedback = st.text_area("Please leave your feedback here:", "")

            # Button to submit feedback
            if st.button("Submit Feedback"):
                insert_feedback(conn, feedback)

            conn.close()

        if st.button("About"):
            st.text("-by Riya")
            st.text("Built with Streamlit")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

