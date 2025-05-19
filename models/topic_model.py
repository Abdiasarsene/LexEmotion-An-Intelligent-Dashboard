from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeler:
    def __init__(self, n_topics=5):
        self.vectorizer = CountVectorizer(max_df=0.95, stop_words="english")
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    def model_topics(self, doc: str):
        """
        doc : un seul document string → transformé en liste pour le vectorizer
        """
        X = self.vectorizer.fit_transform([doc])
        self.lda.fit(X)
        return self.lda, self.vectorizer
