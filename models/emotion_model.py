from transformers import pipeline
from transformers import AutoTokenizer

class EmotionAnalyzer:
    def __init__(self):
        model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
        self.model = pipeline("text-classification", model=model_name, top_k=None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk_text(self, text, max_length=512):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def analyze_emotions(self, text):
        # Si texte trop long → découpe
        if len(self.tokenizer.encode(text)) > 512:
            chunks = self.chunk_text(text)
            results = []
            for chunk in chunks:
                result = self.model(chunk)
                results.append(result)
            # Option : ici tu peux moyenner ou consolider les émotions
            return results
        else:
            return self.model(text)
    