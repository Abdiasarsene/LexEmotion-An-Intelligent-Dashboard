from models.pipeline import NLPPipeline

if __name__ == "__main__":
    # Initialisation du pipeline
    pipeline = NLPPipeline()

    # Fichier ou dossier à traiter
    file_path = r"D:\Projects\IT\Data Science & IA\Detecting emotions in corporate incident reports\data\Rappord d'incidents - Projet 3.pdf"  # adapte selon ton fichier

    # Lancement de l'analyse
    pipeline.run(file_path)
