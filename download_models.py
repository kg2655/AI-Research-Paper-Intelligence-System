import time
from gliner import GLiNER
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

def download_all_models():
    print("Starting model downloads. This will cache the models locally.")
    
    # 1. GLiNER
    print("\n--- Downloading GLiNER (This might take a while) ---")
    GLiNER.from_pretrained('urchade/gliner_medium-v2.1')
    
    # 2. FLAN-T5
    print("\n--- Downloading FLAN-T5 ---")
    T5Tokenizer.from_pretrained('google/flan-t5-base')
    T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    
    # 3. Sentence Transformers
    print("\n--- Downloading Sentence Transformers ---")
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("\n✅ All models downloaded and cached successfully!")

if __name__ == "__main__":
    for attempt in range(1, 6):
        try:
            download_all_models()
            break
        except Exception as e:
            print(f"Network error on attempt {attempt}: {e}")
            if attempt < 5:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Failed to download models after 5 attempts. Please check your internet connection.")
