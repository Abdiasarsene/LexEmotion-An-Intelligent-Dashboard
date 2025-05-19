from models.preprocessing import TextPreprocessor
from models.emotion_model import EmotionAnalyzer
from models.topic_model import TopicModeler
from models.visualization import ResultsVisualizer


class NLPPipeline:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_modeler = TopicModeler()
        self.visualizer = ResultsVisualizer()  # Initialisation du visualiseur

    def run(self, path: str):
        # Étape 1 : Extraction de texte
        text = self.preprocessor.load_pdf(path)

        # Étape 2 : Détection de la langue et traduction si nécessaire
        lang = self.preprocessor.detect_language(text)
        print(f"Langue détectée : {lang}")
        if lang != 'en':
            text = self.preprocessor.translate_to_english(text)
            print("Texte traduit.")

        # Étape 3 : Nettoyage
        cleaned_text = self.preprocessor.clean(text)
        print("Texte nettoyé.")

        # Étape 4 : Analyse des émotions
        emotions = self.emotion_analyzer.analyze_emotions(cleaned_text)
        print("Analyse des émotions terminée.")

        # Étape 5 : Analyse des thématiques
        lda_model, vectorizer = self.topic_modeler.model_topics(cleaned_text)
        print("Modélisation des sujets (LDA) terminée.")

        # Étape 6 : Visualisation
        self.visualizer.plot_emotions(emotions)
        self.visualizer.plot_lda_topics(lda_model, vectorizer)

        # Retour des résultats
        return {
            "cleaned_text": cleaned_text,
            "emotions": emotions,
            "lda_model": lda_model,
            "vectorizer": vectorizer
        }
