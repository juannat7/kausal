from scipy import stats

def hypothesis_testing(
        causal_effect,
        non_causal_effect
    ):
        """
        Perform hypothesis testing between two losses, e.g., causal / non-causal effects.
        Testing whether the cause variable adds value in the effect estimation. 
            H0: diff <= 0; 
            H1: diff > 0
        """
    
        causal_effect = causal_effect.detach().numpy().flatten()
        non_causal_effect = non_causal_effect.detach().numpy().flatten()
        
        return stats.wilcoxon(
            causal_effect - non_causal_effect,
            alternative='greater'
        )
        