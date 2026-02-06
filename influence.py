import torch
import pandas as pd
import pickle
from time import time
from tqdm import tqdm
from collections import defaultdict

class TOFUInfluenceEngine(object):
    """
    Tailored Influence Engine for OLMo-2 on TOFU dataset.
    Focuses on 'Proposed' and 'Identity' methods for scalability.
    """
    def __init__(self):
        self.time_dict = defaultdict(float)
        self.hvp_dict = defaultdict(dict)
        self.IF_dict = {}

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        """
        tr_grad_dict: Gradients from the 200 TOFU biographies.
        val_grad_dict: Gradients from the 'Forget Set' questions.
        """
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict
        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        
        # Ensure gradients are on the same device (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_hvps(self, lambda_const_param=10):
        # Identity is the baseline (simple gradient dot product)
        self.compute_hvp_identity()
        # Proposed is the DataInf optimization
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param)

    def compute_hvp_identity(self):
        start_time = time()
        # Identity assumes the Hessian is just the identity matrix
        self.hvp_dict["identity"] = self.val_grad_dict.copy()
        self.time_dict["identity"] = time() - start_time

    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict = defaultdict(dict)
        
        # We iterate through each 'Forget' question (val_id)
        for val_id in tqdm(self.val_grad_dict.keys(), desc="Computing HVPs"):
            for weight_name in self.val_grad_dict[val_id]:
                # 1. Lambda (damping) computation: prevents division by zero
                # and stabilizes the Hessian approximation for LLM weights.
                S = torch.zeros(self.n_train, device=self.device)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.device)
                    S[tr_id] = torch.mean(tmp_grad**2)
                
                lambda_const = torch.mean(S) / lambda_const_param

                # 2. Influence approximation
                hvp = torch.zeros_like(self.val_grad_dict[val_id][weight_name], device=self.device)
                v = self.val_grad_dict[val_id][weight_name].to(self.device)
                
                for tr_id in self.tr_grad_dict:
                    u = self.tr_grad_dict[tr_id][weight_name].to(self.device)
                    # This is the 'Closed-Form' trick from the DataInf paper:
                    # It approximates (H + ?I)^-1 * v using Sherman-Morrison logic
                    dot_product = torch.sum(v * u)
                    norm_sq = torch.sum(u**2)
                    
                    c_tmp = dot_product / (lambda_const + norm_sq)
                    hvp += (v - c_tmp * u) / (self.n_train * lambda_const)
                
                hvp_proposed_dict[val_id][weight_name] = hvp.cpu() # Move to CPU to save VRAM
                
        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time() - start_time

    def compute_IF(self):
        """
        Calculates the final Influence score: dot product of tr_grad and HVP.
        """
        for method_name in self.hvp_dict:
            print(f"Calculating Influence Scores via {method_name}...")
            # Result: DataFrame where Rows = Train Biographies, Cols = Forget Questions
            influence_matrix = torch.zeros((self.n_train, self.n_val))
            
            for tr_id in range(self.n_train):
                for val_id in range(self.n_val):
                    score = 0
                    for weight_name in self.tr_grad_dict[tr_id]:
                        u = self.tr_grad_dict[tr_id][weight_name]
                        hvp = self.hvp_dict[method_name][val_id][weight_name]
                        score += torch.sum(hvp * u)
                    influence_matrix[tr_id, val_id] = -score
            
            self.IF_dict[method_name] = pd.DataFrame(influence_matrix.numpy())

    def save_result(self, filename="tofu_influence_results.pkl"):
        results = {
            'runtime': dict(self.time_dict),
            'influence': self.IF_dict
        }
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
