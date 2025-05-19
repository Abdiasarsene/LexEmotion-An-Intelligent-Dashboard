import fitz  # PyMuPDF
import spacy
from langdetect import detect
from googletrans import Translator, LANGUAGES

class TextPreprocessor:
    def __init__(self, model="en_core_web_sm"):
        # Charger le modèle spaCy
        self.nlp = spacy.load(model)
        self.translator = Translator()  # Initialiser le traducteur ici

    def load_pdf(self, path: str) -> str:
        """
        Charge un PDF et extrait le texte.
        """
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def detect_language(self, text: str) -> str:
        """
        Détecte la langue du texte.
        """
        return detect(text)

    def translate_to_english(self, text: str) -> str:
        """
        Traduire le texte en anglais si nécessaire.
        """
        try:
            translated = self.translator.translate(text, dest='en')
            if translated.text:
                return translated.text
            else:
                raise ValueError("Traduction échouée : aucun texte retourné.")
        except Exception as e:
            print(f"Erreur lors de la traduction : {e}")
            return text  # Retourner le texte original en cas d'échec

    def clean(self, text: str) -> str:
        """
        Nettoie le texte en retirant les stopwords et la ponctuation,
        puis effectue une lemmatisation.
        """
        doc = self.nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin du fichier PDF
    pdf_path = r"D:\Projects\IT\Data Science & IA\Detecting emotions in corporate incident reports\data\Rappord d'incidents - Projet 3.pdf"  # Remplace ce chemin par ton fichier PDF

    # Créer une instance de la classe TextPreprocessor
    preprocessor = TextPreprocessor()

    # Étape 1: Charger et extraire le texte
    text = preprocessor.load_pdf(pdf_path)
    
    # Étape 2: Détection de la langue
    detected_lang = preprocessor.detect_language(text)
    print(f"Langue détectée : {detected_lang}")
    
    # Étape 3: Traduction automatique si la langue n'est pas l'anglais
    if detected_lang != 'en':
        text = preprocessor.translate_to_english(text)
        print(f"Texte traduit en anglais : {text}")

    # Étape 4: Nettoyage du texte (lemmatisation et suppression des stopwords et ponctuation)
    cleaned_text = preprocessor.clean(text)
    print(f"Texte nettoyé : {cleaned_text}")
