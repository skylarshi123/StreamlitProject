import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Streamlit Classification App")
#/Users/skylarshi/Documents/Streamlit/dataset.csv
# Sidebar options
data = pd.read_csv("/Users/skylarshi/Documents/Streamlit/Personality_test.csv", nrows = 1000)
data = data.drop(columns = ['Unnamed: 0'])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
unique_countries = data['country'].unique()
print(unique_countries)
data['country'] = label_encoder.fit_transform(data['country'])

data = data.dropna()
sample_data = data.sample(frac=0.01, random_state=42)
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
model = AgglomerativeClustering()
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(sample_data)
visualizer.show()
st.write("Choose the number of clusters: pick a number 2 to 10")
n_clusters = st.number_input("Enter the number of clusters", min_value=2, max_value=10)
model = AgglomerativeClustering(n_clusters=n_clusters)
y_model = model.fit_predict(sample_data)
sample_data['cluster'] = y_model
#Now I want the user to input their data to see what cluster they belong to
# EXT1	I am the life of the party.
# EXT2	I don't talk a lot.
# EXT3	I feel comfortable around people.
# EXT4	I keep in the background.
# EXT5	I start conversations.
# EXT6	I have little to say.
# EXT7	I talk to a lot of different people at parties.
# EXT8	I don't like to draw attention to myself.
# EXT9	I don't mind being the center of attention.
# EXT10	I am quiet around strangers.
# EST1	I get stressed out easily.
# EST2	I am relaxed most of the time.
# EST3	I worry about things.
# EST4	I seldom feel blue.
# EST5	I am easily disturbed.
# EST6	I get upset easily.
# EST7	I change my mood a lot.
# EST8	I have frequent mood swings.
# EST9	I get irritated easily.
# EST10	I often feel blue.
# AGR1	I feel little concern for others.
# AGR2	I am interested in people.
# AGR3	I insult people.
# AGR4	I sympathize with others' feelings.
# AGR5	I am not interested in other people's problems.
# AGR6	I have a soft heart.
# AGR7	I am not really interested in others.
# AGR8	I take time out for others.
# AGR9	I feel others' emotions.
# AGR10	I make people feel at ease.
# CSN1	I am always prepared.
# CSN2	I leave my belongings around.
# CSN3	I pay attention to details.
# CSN4	I make a mess of things.
# CSN5	I get chores done right away.
# CSN6	I often forget to put things back in their proper place.
# CSN7	I like order.
# CSN8	I shirk my duties.
# CSN9	I follow a schedule.
# CSN10	I am exacting in my work.
# OPN1	I have a rich vocabulary.
# OPN2	I have difficulty understanding abstract ideas.
# OPN3	I have a vivid imagination.
# OPN4	I am not interested in abstract ideas.
# OPN5	I have excellent ideas.
# OPN6	I do not have a good imagination.
# OPN7	I am quick to understand things.
# OPN8	I use difficult words.
# OPN9	I spend time reflecting on things.
# OPN10	I am full of ideas.
#Country
#I want to have a slider from 0 to 10 for all these questions
#I want to have a drop down menu for country

# Questions
questions = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas."
}

# Countries
countries = [
    "United Kingdom", "Malaysia", "Kenya", "Sweden", "United States", "Finland", "Ukraine", "Philippines", "France", "Australia",
    "India", "Canada", "Netherlands", "South Africa", "Hong Kong", "Brazil", "Switzerland", "Thailand", "Italy", "Spain",
    "United Arab Emirates", "Croatia", "Greece", "Ireland", "Oman", "Germany", "Portugal", "Singapore", "Romania", "Norway",
    "Bangladesh", "Brunei", "Nigeria", "Lithuania", "Ethiopia", "Indonesia", "Belgium", "Austria", "Denmark", "Tanzania",
    "Luxembourg", "Poland", "Japan", "Mexico", "Cyprus", "Uganda", "Sri Lanka", "Turkey", "Myanmar", "Colombia",
    "Estonia", "Argentina", "Iceland", "Hungary", "Puerto Rico", "Pakistan", "Tunisia", "Latvia", "Czech Republic", "New Zealand",
    "Serbia", "Israel", "Jamaica", "Chile", "Qatar", "Saudi Arabia", "Vietnam", "Kazakhstan", "Bosnia and Herzegovina",
    "Mauritius", "Egypt", "Peru", "Slovenia", "Jordan", "Taiwan", "Dominican Republic", "Algeria", "Kuwait", "Morocco",
    "Malta", "Trinidad and Tobago", "Bahamas", "Venezuela", "Russia", "South Korea", "Liberia", "Guatemala", "Bulgaria",
    "Isle of Man", "Ghana", "Somalia", "Slovakia", "Maldives", "China", "Azerbaijan", "Albania", "Cambodia",
    "Lebanon", "Uruguay", "Zimbabwe", "Uzbekistan", "Honduras", "Costa Rica", "Georgia", "Gibraltar", "Macau", "Nepal",
    "Iran", "North Macedonia", "Mongolia", "Zambia", "Nicaragua", "Bahrain", "Sudan", "Belize", "Grenada", "Cayman Islands",
    "Barbados", "Ivory Coast", "Papua New Guinea", "Antigua and Barbuda", "U.S. Virgin Islands", "Paraguay", "Panama", "Eswatini",
    "El Salvador", "Montenegro", "Bermuda", "Angola", "Kyrgyzstan", "Fiji", "Saint Vincent and the Grenadines", "Afghanistan", "Rwanda",
    "Guernsey", "Belarus", "Guadeloupe", "Åland Islands", "Libya", "Jersey", "Northern Mariana Islands", "Syria",
    "Palestine", "Gabon", "Moldova", "Guam", "Armenia", "Ecuador", "British Virgin Islands", "Yemen", "Curaçao", "French Polynesia",
    "Dominica", "Botswana", "Burundi", "Aruba", "Cameroon", "Saint Lucia", "Guyana", "Cape Verde", "Lesotho", "Gambia",
    "Iraq", "Bolivia", "Laos", "Kosovo", "Suriname", "South Sudan", "Cuba", "New Caledonia", "Mozambique", "Senegal",
    "Seychelles", "Faroe Islands", "Malawi", "Palau", "Madagascar", "Niue", "Anguilla", "Saint Kitts and Nevis", "Vanuatu",
    "Monaco", "Cook Islands", "Martinique", "Benin", "Bhutan", "Antarctica", "Greenland", "Montserrat", "Haiti", "Falkland Islands",
    "Democratic Republic of the Congo", "Marshall Islands", "Turks and Caicos Islands", "Réunion", "Mali", "Bonaire",
    "Republic of the Congo", "Mauritania", "Burkina Faso", "Tajikistan", "San Marino", "Andorra", "French Guiana", "Sierra Leone",
    "Comoros", "Liechtenstein", "Togo", "Sint Maarten", "Micronesia", "Tonga", "Samoa", "Timor-Leste", "Equatorial Guinea",
    "Chad", "Saint Martin", "Saint Pierre and Miquelon", "Niger", "Djibouti", "Guinea", "American Samoa", "Saint Helena"
]
# Dropdown for country selection
selected_country = st.selectbox("Select your country:", countries)

# Creating sliders for each question
responses = {}
for key, question in questions.items():
    responses[key] = st.slider(question, 0, 10, 5)


# Here you would add code to process the responses and determine the cluster

# For example:
# cluster = determine_cluster(responses, selected_country)
# st.write("Your cluster:", cluster)
# Convert responses to a DataFrame
user_data = pd.DataFrame([responses])

# Dictionary to convert full country names to two-letter codes
country_code_map = {
    "United States": "US", "Canada": "CA", "United Kingdom": "GB", "Australia": "AU", "Germany": "DE",
    "France": "FR", "Japan": "JP", "China": "CN", "India": "IN", "Brazil": "BR",
    # Add more mappings as needed
}

# Encode the selected country
selected_country_code = country_code_map.get(selected_country, "UNKNOWN")
user_country = label_encoder.transform([selected_country_code])[0]
user_data['country'] = user_country

# Ensure the user data has the same columns as the sample data
user_data = user_data.reindex(columns=sample_data.columns[:-1], fill_value=0)

from scipy.spatial.distance import cdist

# Calculate cluster centroids (do this once after initial clustering)
cluster_centroids = sample_data.groupby('cluster').mean()

# Function to predict cluster for new data
def predict_cluster(new_data):
    distances = cdist(new_data, cluster_centroids, metric='euclidean')
    return distances.argmin()

# Use the function to predict the cluster for the user's data
user_cluster = predict_cluster(user_data)

st.write(f"Based on your responses, you belong to cluster: {user_cluster}")

import numpy as np

# Optional: Display cluster characteristics
cluster_means = sample_data.groupby('cluster').mean()

# Function to get average for a trait
def get_trait_average(data, trait):
    return data[data.index.str.startswith(trait)].mean()

# Calculate averages for each trait
user_cluster_data = cluster_means.loc[user_cluster]
trait_averages = {
    'Extraversion': get_trait_average(user_cluster_data, 'EXT'),
    'Emotional Stability': get_trait_average(user_cluster_data, 'EST'),
    'Agreeableness': get_trait_average(user_cluster_data, 'AGR'),
    'Conscientiousness': get_trait_average(user_cluster_data, 'CSN'),
    'Openness': get_trait_average(user_cluster_data, 'OPN')
}

# Function to interpret score
def interpret_score(score):
    if score < 2:
        return "Very Low"
    elif score < 3:
        return "Low"
    elif score < 4:
        return "Average"
    elif score < 5:
        return "High"
    else:
        return "Very High"

# Display the results with interpretation
st.write("Characteristics of your cluster:")
for trait, avg in trait_averages.items():
    interpretation = interpret_score(avg)
    st.write(f"{trait}: {avg:.2f} - {interpretation}")

# Handle the country
country_code = int(user_cluster_data['country'])
# Create a reverse mapping of encoded values to country codes
encoded_to_code = {label_encoder.transform([code])[0]: code for code in label_encoder.classes_}

# Find the closest country code
closest_encoded = min(encoded_to_code.keys(), key=lambda x: abs(x - country_code))
closest_country_code = encoded_to_code[closest_encoded]

country_name = next((name for name, code in country_code_map.items() if code == closest_country_code), "Unknown")
st.write(f"Most representative country: {country_name}")