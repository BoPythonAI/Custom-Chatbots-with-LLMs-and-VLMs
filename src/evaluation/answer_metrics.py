"""
Answer Quality Evaluation Metrics
Implements Accuracy, BLEU, ROUGE, BERTScore for answer evaluation
"""
import re
from typing import List, Dict, Optional
import numpy as np

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class AnswerEvaluator:
    """Evaluate answer quality using multiple metrics"""
    
    def __init__(self):
        """Initialize answer evaluator"""
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def extract_answer_choice(self, answer_text: str, choices: List[str]) -> Optional[int]:
        """
        Extract answer choice index from answer text
        
        Args:
            answer_text: Generated answer text
            choices: List of choice options
            
        Returns:
            Answer index (0, 1, 2, 3) or None if not found
        """
        if not answer_text or not choices:
            return None
        
        answer_text = answer_text.upper().strip()
        
        # Try to find A/B/C/D at the beginning or after space
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            # Check for "A.", "A)", "A:", or standalone "A"
            if (answer_text.startswith(letter) or 
                f" {letter}." in answer_text or 
                f" {letter})" in answer_text or
                f" {letter}:" in answer_text):
                return i
        
        # Try to find choice text match (partial match)
        for i, choice in enumerate(choices):
            choice_clean = choice.strip().lower()
            answer_lower = answer_text.lower()
            # Check if choice text appears in answer
            if choice_clean in answer_lower or answer_lower in choice_clean:
                return i
        
        return None
    
    def calculate_accuracy(
        self,
        generated_answers: List[str],
        ground_truth_indices: List[int],
        choices_list: List[List[str]]
    ) -> Dict[str, float]:
        """
        Calculate accuracy (exact match for multiple choice)
        
        Args:
            generated_answers: List of generated answer texts
            ground_truth_indices: List of correct answer indices
            choices_list: List of choice options for each question
            
        Returns:
            Dictionary with accuracy metrics
        """
        correct = 0
        total = len(generated_answers)
        correct_predictions = []
        
        for gen_answer, gt_idx, choices in zip(generated_answers, ground_truth_indices, choices_list):
            predicted_idx = self.extract_answer_choice(gen_answer, choices)
            is_correct = (predicted_idx == gt_idx)
            correct_predictions.append(is_correct)
            if is_correct:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def calculate_bleu(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BLEU score
        
        Args:
            generated_answers: List of generated answer texts
            reference_answers: List of reference answer texts
            
        Returns:
            Dictionary with BLEU score
        """
        if not BLEU_AVAILABLE:
            return {"bleu": 0.0}
        
        if not generated_answers or not reference_answers:
            return {"bleu": 0.0}
        
        try:
            bleu = BLEU()
            score = bleu.corpus_score(generated_answers, [reference_answers])
            return {"bleu": score.score / 100.0}  # Convert to 0-1 scale
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
            return {"bleu": 0.0}
    
    def calculate_rouge(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            generated_answers: List of generated answer texts
            reference_answers: List of reference answer texts
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        if not generated_answers or not reference_answers:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            if not gen or not ref:
                continue
            try:
                scores = self.rouge_scorer.score(ref, gen)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                continue
        
        if not rouge1_scores:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores)
        }
    
    def calculate_bertscore(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        Calculate BERTScore
        
        Args:
            generated_answers: List of generated answer texts
            reference_answers: List of reference answer texts
            lang: Language code
            
        Returns:
            Dictionary with BERTScore (precision, recall, F1)
        """
        if not BERTSCORE_AVAILABLE:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        if not generated_answers or not reference_answers:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        try:
            P, R, F1 = bert_score(generated_answers, reference_answers, lang=lang, verbose=False)
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def evaluate_all(
        self,
        generated_answers: List[str],
        ground_truth_indices: List[int],
        choices_list: List[List[str]],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            generated_answers: List of generated answer texts
            ground_truth_indices: List of correct answer indices
            choices_list: List of choice options
            reference_texts: List of reference answer texts (for BLEU/ROUGE/BERTScore)
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Accuracy
        accuracy_metrics = self.calculate_accuracy(generated_answers, ground_truth_indices, choices_list)
        results.update(accuracy_metrics)
        
        # BLEU, ROUGE, BERTScore (if reference texts provided)
        if reference_texts:
            bleu_scores = self.calculate_bleu(generated_answers, reference_texts)
            results.update(bleu_scores)
            
            rouge_scores = self.calculate_rouge(generated_answers, reference_texts)
            results.update(rouge_scores)
            
            bert_scores = self.calculate_bertscore(generated_answers, reference_texts)
            results.update(bert_scores)
        
        return results

