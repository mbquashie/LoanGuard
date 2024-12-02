# utils/feature_mapping.py

FEATURE_MAPPING = {
    'avg_cur_bal': 'Average Current Balance',
    'mo_sin_old_rev_tl_op': 'Months Since Oldest Revolving Trade Line Open',
    'num_accts_ever_120_pd': 'Number of Accounts Ever 120 Days Past Due',
    'num_actv_rev_tl': 'Number of Active Revolving Trade Lines',
    'num_bc_tl': 'Number of Bankcard Trade Lines',
    'num_il_tl': 'Number of Installment Trade Lines',
    'num_tl_op_past_12m': 'Number of Trade Lines Open Past 12 Months',
    'pub_rec_bankruptcies': 'Number of Public Record Bankruptcies',
    'tax_liens': 'Number of Tax Liens',
    'tot_coll_amt': 'Total Collection Amount',
    'total_il_high_credit_limit': 'Total Installment High Credit Limit'
}

# Create reverse mapping for potential future use
REVERSE_FEATURE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}

