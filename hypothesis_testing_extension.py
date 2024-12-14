import json
import numpy as np
from scipy import stats


if __name__ == "__main__":

    # probs GEIA
    with open("./logs/leakage_attacker_rand_gpt2_m_personachat_sent_roberta_beam.log") as f:
        probs = json.load(f)

    original_mean = np.mean(probs["positive_sample_likelihood"])
    negative_mean = np.mean(probs["negative_sample_likelihood"])
    original_std = np.std(probs["positive_sample_likelihood"])
    negative_std = np.std(probs["negative_sample_likelihood"])

    print(f"Original mean: {original_mean} with std: {original_std}\nNegative mean: {negative_mean} with std: {negative_std}")

    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(probs["positive_sample_likelihood"], probs["negative_sample_likelihood"])

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
            print("Reject the null hypothesis; there is a significant difference between the distributions.")
    else:
        print("Fail to reject the null hypothesis; there is no significant difference between the distributions.")