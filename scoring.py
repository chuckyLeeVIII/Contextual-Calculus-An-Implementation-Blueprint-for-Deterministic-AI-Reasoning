import numpy as np
from datetime import datetime

class ContextualCalculus:
    def __init__(self, weights={'w_s': 0.5, 'w_t': 0.3, 'w_e': 0.2}):
        self.weights = weights
        self.current_year = 2026

    def calculate_sas(self, source_type):
        scores = {
            'persona_work': 1.0,
            'transcript': 0.9,
            'correspondence': 0.8,
            'biographical_fact': 0.7,
            'associate_statement': 0.5,
            'neutral_party': 0.2,
            'critic': -0.3
        }
        return scores.get(source_type, 0.0)

    def calculate_trs(self, chunk_year):
        trs = 1 - ((self.current_year - chunk_year) / 100)
        return max(0, min(1, trs))

    def calculate_ecs(self, query_sentiment, chunk_sentiment):
        # Cosine similarity simplified
        dot_product = np.dot(query_sentiment, chunk_sentiment)
        norm_q = np.linalg.norm(query_sentiment)
        norm_c = np.linalg.norm(chunk_sentiment)
        if norm_q == 0 or norm_c == 0:
            return 0
        return dot_product / (norm_q * norm_c)

    def compute_fvw(self, sas, trs, ecs, rs):
        weighted_sum = (self.weights['w_s'] * sas) + \
                       (self.weights['w_t'] * trs) + \
                       (self.weights['w_e'] * ecs)
        return weighted_sum * rs

# Apply Theory Deterministically to Comet's approach:
# 1. Break down: SAS (Source Authority), TRS (Temporal), ECS (Emotional)
# 2. Deterministic: Mathematical weighting vs probabilistic guessing
# 3. Rebuild: Scoring engine for grounding responses in verified context

def apply_to_self():
    engine = ContextualCalculus()
    
    # Example: User query about this repo
    query_sent = [0.1, 0.0, 0.9] # Neutral/Curious
    
    # Evidence from README (SAS=1.0, Date=2025, RS=0.95)
    chunk_sent = [0.2, 0.0, 0.8]
    sas = engine.calculate_sas('persona_work')
    trs = engine.calculate_trs(2025)
    ecs = engine.calculate_ecs(query_sent, chunk_sent)
    rs = 0.95
    
    fvw = engine.compute_fvw(sas, trs, ecs, rs)
    print(f"Deterministic Reasoning Weight for README chunk: {fvw:.4f}")

if __name__ == "__main__":
    apply_to_self()
