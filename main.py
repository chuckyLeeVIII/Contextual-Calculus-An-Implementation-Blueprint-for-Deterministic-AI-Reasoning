#!/usr/bin/env python3
# ARCHON INITIALIZED // ADMIN_LEVEL_ACCESS
import os
import json
import numpy as np
from vector_db import ReasoningVDB
from scoring import ContextualCalculus
from video_processor import VideoIngestionEngine

class ContextualAgent:
    def __init__(self, db_path="./vdb_storage"):
        self.vdb = ReasoningVDB(db_path)
        self.calculus = ContextualCalculus(weights={'w_s': 0.6, 'w_t': 0.2, 'w_e': 0.2})
        self.video_engine = VideoIngestionEngine()
        
    def ingest_video(self, video_path, source_type='persona_work', year=2026):
        """Processes a video file, extracts transcripts, and pushes to Vector DB with Contextual metadata."""
        print(f"[*] Ingesting video target: {video_path}")
        chunks = self.video_engine.process_video(video_path)
        
        records = []
        for chunk in chunks:
            records.append({
                'text': chunk['text'],
                'source_type': source_type,
                'year': year,
                'sentiment': chunk['sentiment']
            })
            
        self.vdb.insert_records(records)
        print("[+] Video ingestion and vectorization complete.")

    def query(self, user_query):
        """Runs the deterministic reasoning loop."""
        print(f"[*] Executing query: {user_query}")
        
        # 1. Analyze query sentiment
        query_sentiment = self.video_engine.analyze_sentiment(user_query)
        
        # 2. Vector DB Retrieval
        raw_results = self.vdb.search(user_query, top_k=10)
        
        # 3. Apply Contextual Calculus
        scored_results = []
        for res in raw_results:
            sas = self.calculus.calculate_sas(res['metadata']['source_type'])
            trs = self.calculus.calculate_trs(res['metadata']['year'])
            ecs = self.calculus.calculate_ecs(query_sentiment, res['metadata']['sentiment'])
            rs = res['score'] # Vector similarity
            
            fvw = self.calculus.compute_fvw(sas, trs, ecs, rs)
            res['fvw'] = fvw
            scored_results.append(res)
            
        # 4. Sort by Final Variable Weight
        scored_results.sort(key=lambda x: x['fvw'], reverse=True)
        
        # 5. Construct Deterministic Prompt
        context_string = ""
        for i, res in enumerate(scored_results[:5]):
            context_string += f"[FVW: {res['fvw']:.4f}] Chunk {i} (Source: {res['metadata']['source_type']}, {res['metadata']['year']}): \\"{res['text']}\\"\\n"
            
        prompt = f"""
        **Persona Mandate:** You are an advanced reasoning AI. Synthesize the provided Contextual Evidence to answer the user's query. You MUST ground your response exclusively in the evidence provided. Prioritize higher-weighted evidence. Do not use outside knowledge.
        
        **User Query:** "{user_query}"
        
        **Contextual Evidence (Ranked by FVW):**
        {context_string}
        """
        
        print("[+] Constructed Deterministic Prompt. Ready for LLM execution.")
        return prompt

if __name__ == "__main__":
    agent = ContextualAgent()
    # Ingest the video data as mandated
    # agent.ingest_video("raw_video_footage.mp4") 
    
    # Execute Contextual Calculus vector pipeline
    prompt = agent.query("Expand the theory with the vector database mechanics and the video integration.")
    print(prompt)
