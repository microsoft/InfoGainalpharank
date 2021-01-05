import numpy as np
import logging


def run_sampling(payoff_matrix_sampler, sampler, max_iters=100, graph_samples=10, true_alpha_rank=None, true_payoff=None):
    logger = logging.getLogger("Sampling")
    logger.warning("Starting sampling for up to {:,} iterations.".format(max_iters))
    
    # Quantities to log
    improvements = []
    entries = []
    alpha_ranking_distrib = []
    alpha_ranking_distrib_more = []
    mean_alpha_ranks = []
    prob_alpha_ranks = []
    prob_alpha_rank_ps = []
    payoff_matrix_means = []
    payoff_matrix_vars = []

    for t in range(max_iters):
        # Log current (approximation) of distribution over alpha-rankings
        if t % (max_iters//100) == 0:
            logger.warning("Iteration {:,}".format(t))
            # Periodically get a better approximation of the distribution
            alpha_rankings_distrib_samples, prob_alpha_rank = sampler.alpha_rankings_distrib(graph_samples=graph_samples)
            alpha_ranking_distrib_more.append(alpha_rankings_distrib_samples)

            if isinstance(prob_alpha_rank, tuple):
                # We have the most probable alpha rank and its probability
                prob_phi, prob_phi_p = prob_alpha_rank
                prob_alpha_rank_ps.append(prob_phi_p)
                if true_alpha_rank is not None:
                    logger.info("L1 Error to most probable: {:.2f}. Probability of {:.4f}".format(np.abs(prob_phi - true_alpha_rank).sum(), prob_phi_p))
            else:
                prob_phi = prob_alpha_rank
                if true_alpha_rank is not None:
                    logger.info("L1 Error to most probable: {:.2f}".format(np.abs(prob_phi - true_alpha_rank).sum()))
            prob_alpha_ranks.append(prob_phi)

        logger.debug("Iteration {:,}".format(t))

        # Log the mean alpha rank (alpha rank of mean payoff estimates)
        mean_alpha_rank = sampler.alpha_rankings_distrib(mean=True)
        mean_alpha_ranks.append(mean_alpha_rank)
        if true_alpha_rank is not None:
            logger.info("L1 Error: {:.2f}".format(np.abs(mean_alpha_rank - true_alpha_rank).sum()))

        # Log current means and variances for each entry of the payoff matrix for smaller environments
        if max_iters < 10 * 1000:
            m_means, m_vars = sampler.payoff_distrib()
            payoff_matrix_means.append(m_means)
            payoff_matrix_vars.append(m_vars)
        
        # Pick an entry to sample
        entry_to_sample, sampler_stats = sampler.choose_entry_to_sample()
        if entry_to_sample is None:
            logger.info("Finished sampling at {} iterations".format(t))
            break
        entries.append(entry_to_sample)
        if "improvements" in sampler_stats:
            improvements.append(sampler_stats["improvements"])
        logger.info("Sampling {}".format(entry_to_sample))
        
        # Get a sample from that entry
        payoff_samples = payoff_matrix_sampler.get_entry_sample(entry_to_sample)
        logger.info("Received Payoff {} for {}".format(payoff_samples, entry_to_sample))

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)

    logger.critical("Finished {} iterations".format(max_iters))
    del sampler

    return {
        "improvements": improvements,
        "entries": entries,
        "mean_alpha_rankings": mean_alpha_ranks,
        "prob_alpha_rankings": prob_alpha_ranks,
        "prob_alpha_rank_ps": prob_alpha_rank_ps,
        "alpha_rankings": alpha_ranking_distrib,
        "alpha_rankings_more": alpha_ranking_distrib_more,
        "payoff_matrix_means": payoff_matrix_means,
        "payoff_matrix_vars": payoff_matrix_vars,
    }
