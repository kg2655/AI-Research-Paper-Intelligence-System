from gliner import GLiNER

class TechnicalNER:
    def __init__(self, model_name="urchade/gliner_medium-v2.1"):
        # GLiNER is an extremely capable zero-shot NER model
        self.model = GLiNER.from_pretrained(model_name)
        
        # We can dynamically ask for these labels without any hardcoded dictionary
        self.labels = [
            "Machine Learning Model", 
            "Framework", 
            "Library", 
            "Dataset", 
            "Metric", 
            "Optimizer", 
            "Hardware",
            "Algorithm"
        ]

    def extract_entities(self, text: str) -> dict:
        """
        Extract entities based on the zero-shot labels.
        Returns a dictionary grouped by label.
        """
        # Truncate text for performance or batch it; here we just use the first chunk
        text_to_process = text[:5000] 
        
        entities = self.model.predict_entities(text_to_process, self.labels)
        
        grouped_entities = {}
        for ent in entities:
            label = ent["label"]
            word = ent["text"]
            if label not in grouped_entities:
                grouped_entities[label] = set()
            grouped_entities[label].add(word)
            
        # Convert sets to lists
        return {k: list(v) for k, v in grouped_entities.items()}

    def extract_hyperparameters(self, text: str) -> dict:
        """
        Zero-shot extract hyperparameters using GLiNER.
        """
        hp_labels = [
            "Batch Size",
            "Learning Rate",
            "Epochs",
            "Dropout Rate"
        ]
        
        entities = self.model.predict_entities(text[:5000], hp_labels)
        
        hyperparameters = {}
        for ent in entities:
            hyperparameters[ent["label"]] = ent["text"]
            
        return hyperparameters
