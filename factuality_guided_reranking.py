"""
Factuality-Guided Re-ranking (FGR) System
=========================================
Novel Contribution: Multi-model ensemble with factuality-aware sentence selection

Date: January 2025

This is the PROPOSED MODEL for the paper.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LEDTokenizer, LEDForConditionalGeneration
import torch
import numpy as np
from rouge_score import rouge_scorer
from factuality_module import FactualityChecker
import nltk
from pulp import *
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


class FactualityGuidedReranker:
    """
    Novel Model: FGR (Factuality-Guided Re-ranking)
    
    Combines:
    1. Multi-model generation (LED + BART + DistilBART)
    2. Factuality scoring (entity, temporal, semantic)
    3. Optimal sentence selection (ILP)
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print("\n" + "="*80)
        print("INITIALIZING FGR (FACTUALITY-GUIDED RE-RANKING)")
        print("="*80)
        
        # Load 3 best baseline models
        print("\n Loading Model 1: LED-ArXiv...")
        self.led_tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384-arxiv')
        self.led_model = LEDForConditionalGeneration.from_pretrained('allenai/led-large-16384-arxiv')
        self.led_model = self.led_model.to(self.device)
        self.led_model.eval()
        
        print(" Loading Model 2: BART-Base...")
        self.bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
        self.bart_model = self.bart_model.to(self.device)
        self.bart_model.eval()
        
        print(" Loading Model 3: DistilBART...")
        self.distilbart_tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.distilbart_model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.distilbart_model = self.distilbart_model.to(self.device)
        self.distilbart_model.eval()
        
        # Factuality checker
        print("\n Loading Factuality Checker...")
        self.fact_checker = FactualityChecker()
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print("\n" + "="*80)
        print("FGR SYSTEM READY")
        print("="*80 + "\n")
    
    def generate_candidate_summaries(self, article, num_beams=4, max_length=150):
        """
        Stage 1: Generate summaries from 3 models
        """
        candidates = []
        
        # LED summary
        with torch.no_grad():
            inputs = self.led_tokenizer(article, max_length=16384, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.led_model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
            led_summary = self.led_tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(('LED', led_summary))
        
        # BART summary
        with torch.no_grad():
            inputs = self.bart_tokenizer(article, max_length=1024, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.bart_model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
            bart_summary = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(('BART', bart_summary))
        
        # DistilBART summary
        with torch.no_grad():
            inputs = self.distilbart_tokenizer(article, max_length=1024, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.distilbart_model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
            distilbart_summary = self.distilbart_tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(('DistilBART', distilbart_summary))
        
        return candidates
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def score_sentence_factuality(self, sentence, source):
        """
        Score a single sentence for factuality
        """
        scores = self.fact_checker.compute_factuality_score(sentence, source)
        
        # Weighted combination
        factuality = (
            0.4 * scores['entity'] +
            0.3 * scores['temporal'] +
            0.3 * scores['semantic']
        )
        
        return factuality
    
    def score_sentence_rouge(self, sentence, reference):
        """
        Score a single sentence against reference
        """
        scores = self.rouge_scorer.score(reference, sentence)
        # Use ROUGE-2 as main metric
        return scores['rouge2'].fmeasure
    
    def select_sentences_ilp(self, sentences, scores, max_length=150, max_sentences=5):
        """
        Stage 4: Optimal sentence selection using Integer Linear Programming
        
        Maximize: factuality_score + rouge_score
        Subject to: total_length <= max_length
        """
        n = len(sentences)
        
        # Create ILP problem
        prob = LpProblem("Sentence_Selection", LpMaximize)
        
        # Decision variables (binary: select sentence i or not)
        x = [LpVariable(f"x{i}", cat='Binary') for i in range(n)]
        
        # Objective function: maximize total score
        prob += lpSum([scores[i] * x[i] for i in range(n)])
        
        # Constraint 1: Length limit
        lengths = [len(sent.split()) for sent in sentences]
        prob += lpSum([lengths[i] * x[i] for i in range(n)]) <= max_length
        
        # Constraint 2: Maximum number of sentences
        prob += lpSum([x[i] for i in range(n)]) <= max_sentences
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract selected sentences
        selected_indices = [i for i in range(n) if x[i].varValue == 1]
        selected_sentences = [sentences[i] for i in selected_indices]
        
        return selected_sentences, selected_indices
    
    def generate_fgr_summary(self, article, reference=None, verbose=False):
        """
        Complete FGR pipeline
        
        Returns:
            - final_summary: FGR-generated summary
            - metadata: Details about the process
        """
        if verbose:
            print("\n" + "="*80)
            print("FGR GENERATION PIPELINE")
            print("="*80)
        
        # Stage 1: Generate candidate summaries
        if verbose:
            print("\n Stage 1: Generating candidate summaries...")
        candidates = self.generate_candidate_summaries(article)
        
        if verbose:
            for model_name, summary in candidates:
                print(f"  {model_name}: {len(summary.split())} words")
        
        # Stage 2: Extract all sentences
        if verbose:
            print("\n Stage 2: Extracting sentences...")
        all_sentences = []
        for model_name, summary in candidates:
            sents = self.extract_sentences(summary)
            for sent in sents:
                all_sentences.append((model_name, sent))
        
        # Remove duplicates (similar sentences)
        unique_sentences = []
        unique_texts = []
        for model_name, sent in all_sentences:
            # Simple duplicate removal
            if sent not in unique_texts and len(sent.split()) >= 5:
                unique_sentences.append((model_name, sent))
                unique_texts.append(sent)
        
        if verbose:
            print(f"  Total sentences: {len(all_sentences)}")
            print(f"  Unique sentences: {len(unique_sentences)}")
        
        # Stage 3: Score each sentence
        if verbose:
            print("\n Stage 3: Scoring sentences...")
        sentence_scores = []
        
        for model_name, sent in unique_sentences:
            # Factuality score
            fact_score = self.score_sentence_factuality(sent, article)
            
            # ROUGE score (if reference provided)
            if reference:
                rouge_score = self.score_sentence_rouge(sent, reference)
            else:
                rouge_score = 0.5  # Neutral score if no reference
            
            # Combined score (60% factuality + 40% ROUGE)
            combined_score = 0.6 * fact_score + 0.4 * rouge_score
            
            sentence_scores.append({
                'model': model_name,
                'sentence': sent,
                'factuality': fact_score,
                'rouge': rouge_score,
                'combined': combined_score
            })
        
        if verbose:
            print(f"  Average factuality: {np.mean([s['factuality'] for s in sentence_scores]):.3f}")
            print(f"  Average ROUGE: {np.mean([s['rouge'] for s in sentence_scores]):.3f}")
        
        # Stage 4: Optimal selection
        if verbose:
            print("\n Stage 4: Optimal sentence selection (ILP)...")
        
        sentences_only = [s['sentence'] for s in sentence_scores]
        scores_only = [s['combined'] for s in sentence_scores]
        
        selected_sentences, selected_indices = self.select_sentences_ilp(
            sentences_only, 
            scores_only,
            max_length=150,
            max_sentences=5
        )
        
        if verbose:
            print(f"  Selected: {len(selected_sentences)} sentences")
        
        # Create final summary
        final_summary = ' '.join(selected_sentences)
        
        # Metadata
        metadata = {
            'num_candidates': len(candidates),
            'num_sentences_extracted': len(all_sentences),
            'num_unique_sentences': len(unique_sentences),
            'num_selected': len(selected_sentences),
            'selected_indices': selected_indices,
            'selected_scores': [sentence_scores[i] for i in selected_indices]
        }
        
        if verbose:
            print("\n FGR summary generated!")
            print("="*80 + "\n")
        
        return final_summary, metadata


def test_fgr():
    """Test FGR system"""
    print("\n" + "="*80)
    print("TESTING FGR SYSTEM")
    print("="*80)
    
    # Sample article
    article = """
    Prime Minister Narendra Modi announced an economic package worth Rs 20 lakh crore 
    on May 12, 2020, to help India fight COVID-19. The package amounts to 10% of India's GDP.
    Finance Minister Nirmala Sitharaman provided details of the package over five days.
    The package includes measures for MSMEs, poor people, agriculture, and more.
    """
    
    reference = """
    PM Modi announced Rs 20 lakh crore economic package on May 12, 2020. 
    The package is 10% of GDP. FM Sitharaman detailed it over five days.
    """
    
    # Initialize FGR
    fgr = FactualityGuidedReranker(device='cpu')
    
    # Generate summary
    print("\n Generating FGR summary...")
    summary, metadata = fgr.generate_fgr_summary(article, reference, verbose=True)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nArticle ({len(article.split())} words):")
    print(article.strip()[:200] + "...")
    
    print(f"\nReference ({len(reference.split())} words):")
    print(reference.strip())
    
    print(f"\nFGR Summary ({len(summary.split())} words):")
    print(summary)
    
    print(f"\nMetadata:")
    print(f"  Candidates: {metadata['num_candidates']}")
    print(f"  Total sentences: {metadata['num_sentences_extracted']}")
    print(f"  Unique sentences: {metadata['num_unique_sentences']}")
    print(f"  Selected: {metadata['num_selected']}")
    
    # Evaluate factuality
    fact_checker = FactualityChecker()
    fact_scores = fact_checker.compute_factuality_score(summary, article)
    
    print(f"\nFactuality Scores:")
    print(f"  Entity: {fact_scores['entity']}")
    print(f"  Temporal: {fact_scores['temporal']}")
    print(f"  Semantic: {fact_scores['semantic']}")
    print(f"  Overall: {fact_scores['overall']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_fgr()