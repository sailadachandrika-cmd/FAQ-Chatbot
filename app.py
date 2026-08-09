import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
data = pd.read_csv("faq.csv")

questions = data["Question"]
answers = data["Answer"]

# Convert questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("===== FAQ CHATBOT =====")
print("Type 'exit' to stop.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    if similarity[0][index] > 0.2:
        print("Bot:", answers[index])
    else:
        print("Bot: Sorry, I don't know the answer.")