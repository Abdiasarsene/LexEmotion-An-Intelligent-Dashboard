import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ResultsVisualizer:
    def __init__(self):
        sns.set(style="whitegrid")  # Pour un style épuré

    def plot_emotions(self, emotion_scores):
        """
        Affiche un graphique des émotions avec leur score.
        `emotion_scores` doit être une liste de dictionnaires contenant
        les scores de chaque émotion, tel que retourné par Hugging Face.
        """
        if not emotion_scores:
            print("Aucune émotion à afficher.")
            return

        # Agrégation des scores par label
        emotion_totals = {}
        for emotion_set in emotion_scores:
            for item in emotion_set:
                label = item["label"]
                score = item["score"]
                emotion_totals[label] = emotion_totals.get(label, 0) + score

        # Transformation en DataFrame
        df = pd.DataFrame(list(emotion_totals.items()), columns=["Emotion", "Score"])
        df = df.sort_values(by="Score", ascending=False)

        # Tracé
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Score", y="Emotion", data=df, palette="coolwarm")
        plt.title("Distribution des émotions détectées", fontsize=16)
        plt.xlabel("Score total")
        plt.ylabel("Émotion")
        plt.tight_layout()
        plt.show()

    def plot_lda_topics(self, lda_model, vectorizer, n_top_words=10):
        """
        Affiche les n mots les plus représentatifs pour chaque sujet LDA.
        """
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []

        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_keywords.append(top_features)

        # Préparation DataFrame
        topic_df = pd.DataFrame(topic_keywords)
        topic_df.index = [f"Topic {i+1}" for i in range(len(topic_keywords))]

        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(topic_df.apply(lambda x: [1]*len(x), axis=1), 
                    annot=topic_df, fmt="", cmap="Blues", cbar=False,
                    linewidths=.5, linecolor="gray", xticklabels=False)
        plt.title("Top mots par sujet (LDA)", fontsize=16)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
