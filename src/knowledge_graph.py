import networkx as nx
import spacy

class KnowledgeGraphBuilder:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Need to download spacy model in environment
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def build_graph(self, text_chunks: list, extracted_entities: dict) -> nx.MultiDiGraph:
        """
        Builds a knowledge graph using sentence co-occurrence as a proxy for relationships.
        This prevents the 'everything connected to everything' issue of the previous naive approach.
        """
        g = nx.MultiDiGraph()
        
        # Build a flat set of all entities we care about for quick lookup
        all_entities = {}
        for label, items in extracted_entities.items():
            for item in items:
                all_entities[item.lower()] = label

        # Iterate through chunks and sentences to find co-occurring entities
        for chunk in text_chunks:
            doc = self.nlp(chunk)
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Find which entities exist in this sentence
                present_entities = []
                for ent, label in all_entities.items():
                    if ent in sent_text:
                        present_entities.append((ent, label))
                
                # If we have multiple entities in a sentence, connect them
                if len(present_entities) > 1:
                    for i in range(len(present_entities)):
                        for j in range(i + 1, len(present_entities)):
                            ent1, label1 = present_entities[i]
                            ent2, label2 = present_entities[j]
                            
                            # Define basic relationship based on types
                            relation = "co-occurs_with"
                            if label1 == "Machine Learning Model" and label2 == "Dataset":
                                relation = "trained_on"
                            elif label1 == "Machine Learning Model" and label2 == "Framework":
                                relation = "implemented_in"
                            elif label1 == "Machine Learning Model" and label2 == "Metric":
                                relation = "evaluated_by"
                                
                            g.add_edge(ent1, ent2, relation=relation)
                            
        return g

