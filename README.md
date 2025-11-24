# Music_Recommendation_System_Using_MachineLearning

## 🎵 Music Recommendation System using TF-IDF & Cosine Similarity
A Content-Based Music Recommendation Engine built with Python.

## 📌 Project Overview
This project is a Content-Based Music Recommendation System that recommends songs based on textual similarity between features like:
🎼 Track Name
🧑‍🎤 Artist Name
🎧 Genre
Using TF-IDF Vectorization and Cosine Similarity, the system identifies how close two songs are based on their combined textual metadata and suggests the most similar ones. This helps users discover songs that match their taste based on a particular input track.

## 🚀 Features
✔ Visualizes Top 10 Genres

✔ Visualizes Top 10 Artists

✔ Builds a TF-IDF feature matrix

✔ Computes cosine similarity between all songs

✔ Provides top N recommendations for any track

✔ Displays similarity scores

✔ Clean and well-structured code

✔ Fully customizable

## 🛠️ Tech Stack
| Component  | Tools Used                                       |
| ---------- | ------------------------------------------------ |
| Language   | Python                                           |
| Libraries  | Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn |
| Algorithms | TF-IDF Vectorization, Cosine Similarity          |
| Data       | Music Metadata Dataset                           |

## 📂 Dataset Information
The dataset contains fields such as:
* track_name
* artist_name
* genre
* Additional metadata

## 📸 Visualizations
### 🎤 Top 10 Genres
The system plots a bar chart showing the most popular music genres.

### 🎙️ Top 10 Artists
A bar chart showing artists with the highest number of songs in the dataset.

These visualizations help understand the dataset distribution before applying ML techniques.

## 🧠 How The Recommendation System Works
### 1️⃣ Combine Useful Text Features

We merge genre, artist name, and track name into a single feature:
```python
data['combined_features'] = (data['genre'] + ' ' + data['artist_name'] + ' ' + data['track_name'])
```
TF-IDF converts text into mathematical vectors based on word importance.

### 2️⃣ Convert Text to Numbers (TF-IDF)
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])
```
TF-IDF converts text into mathematical vectors based on word importance.

### 3️⃣ Compute Similarity
```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```
Cosine similarity determines how similar two songs are.

### 4️⃣ Get Recommendations
```python
def get_recommendations(song_title, data, cosine_sim, top_n=10):
```
It:
✔ Finds the given song
✔ Retrieves similarity scores
✔ Sorts them
✔ Returns the most similar tracks

## 📝 Sample Results
If you input: cry
You get a list of 10 similar recommended tracks with:
Track Name
Artist
Genre
Similarity Score

## 📊 Recommendation Visualization
A horizontal bar plot shows the recommended song names vs. similarity score for clarity:
```python
sns.barplot(y='track_name', x='similarity', data=recommended_songs)
```
## 📌 Future Improvements
🔹 Add audio-based features (MFCC, tempo, etc.)
🔹 Add user-based collaborative filtering
🔹 Create a Streamlit web UI
🔹 Deploy as an API

## 🤝 Contributions
Pull requests are welcome!
Feel free to open an issue for suggestions or bugs.

## ⭐ Show Your Support
If you like this project, don't forget to ⭐ star the repository on GitHub!







