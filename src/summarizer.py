import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PaperSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Token-aware splitter for T5 max context
        # T5-base usually handles 512 tokens comfortably, we'll keep chunks well within limits
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, # Approx characters
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def summarize_text(self, text: str, max_new_tokens: int = 150) -> str:
        prompt = f"Summarize the following research paper text in detail:\n\n{text}"
        return self._generate(prompt, max_new_tokens)

    def summarize_long_document(self, full_text: str) -> str:
        """Handles documents larger than the model's context window by chunking and combining."""
        chunks = self.splitter.split_text(full_text)
        
        if not chunks:
            return ""
            
        # Summarize first few chunks to save time (often sufficient for paper overview)
        # In a real production system, you might map-reduce over all chunks
        partials = []
        for chunk in chunks[:4]:
            partials.append(self.summarize_text(chunk, max_new_tokens=150))
            
        combined = " ".join(partials)
        
        # Final summary of summaries
        if len(combined.split()) > 150:
            return self.summarize_text(combined, max_new_tokens=220)
        return combined

