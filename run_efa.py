from utils_project import *

#--------------------------------------------------------------------------------------------------------------------
# load data
#--------------------------------------------------------------------------------------------------------------------

print('LOADING DATA')
print('-------------------------------')

# initial sample
init_df  = sample_dict['Initial']['data']
init_dir = f'{base_dir}/data/Initial_2021/Summary'
val_df   = sample_dict['Validation']['data']
val_dir  = f'{base_dir}/data/Replication_2022/Summary/'

# sample sizes
print(f'Initial    n={init_df.shape[0]}')
print(f'Validation n={val_df.shape[0]}')
print(f'Combined   n={val_df.shape[0] + init_df.shape[0]}')

# get questionnaire data & correlation matrices
init_ques_df = subset_df(init_df, fa_prefixes)
val_ques_df  = subset_df(val_df, fa_prefixes)
init_corrmat = init_ques_df.iloc[:,1:].corr()
val_corrmat  = val_ques_df.iloc[:,1:].corr()

# init_corrmat_r = pd.read_csv(f'{init_dir}/Individual_summaries/Questionnaire_hetcor_n636.csv') # corr mat for factor analyis
# init_corrmat_r.set_index('Unnamed: 0', inplace=True)
# val_corrmat_r = pd.read_csv(f'{val_dir}/Individual_summaries/Questionnaire_hetcor_n276.csv')
# val_corrmat_r.set_index('Unnamed: 0', inplace=True)
# init_corrmat_r = init_corrmat_r.loc[init_ques_df.columns[1:], init_ques_df.columns[1:]]
# val_corrmat_r  = val_corrmat_r.loc[val_ques_df.columns[1:], val_ques_df.columns[1:]]

assert np.all(init_ques_df.columns[1:] == val_ques_df.columns[1:]), \
    'initial & validation questionnaire dataframes are disordered'
assert np.all(init_corrmat.columns == init_ques_df.columns[1:]), \
    'initial corr mat & questionnaire dataframe are disordered'
assert np.all(val_corrmat.columns == val_ques_df.columns[1:]), \
    'validation corr mat & questionnaire dataframe are disordered'

#--------------------------------------------------------------------------------------------------------------------
# run factor analyses
#--------------------------------------------------------------------------------------------------------------------

print('\nRUNNING EFA')
print('-------------------------------')

n_comps = 3 
loading_threshs = [25, 50, 75]
rotations = ['quartimax']

for rotation, loading_thresh in itertools.product(rotations, loading_threshs):
    
    #------------------------------------------------------
    # run full factor analysis
    #------------------------------------------------------

    print(f'Running full EFA with {n_comps} components and {rotation} rotation\n')

    # fit to & transform initial sample, transform validation sample
    fa, f_labels, f_summary = run_fa(init_ques_df, init_corrmat, n_comps, rotation)
    init_df[f_labels] = fa.X_reduced
    val_df[f_labels]  = fa.transform(val_ques_df.iloc[:,1:].values)

    # pickle the fa dictionary
    fa_dict = fa.output()
    fa_dict['factor_labels'] = f_summary
    pickle_file(fa_dict, f'{base_dir}/factor_analyses/fa_{rotation}.pkl')

    #------------------------------------------------------
    # run reduced factor analysis
    #------------------------------------------------------

    print(f'\nRunning reduced FA with {loading_thresh}% threshold\n')

    # get efa's best items (largest loadings above some percentile threshold)
    min_items = flatten_lists(fa.efa_partial(thresh=loading_thresh, plot=False))
    min_items.sort()
    init_ques_df_min, init_corrmat_min = init_ques_df.loc[:, min_items], init_corrmat.loc[min_items, min_items]
    assert np.all(init_corrmat_min.columns == init_ques_df_min.columns), \
        'initial reduced corr. matrix & questionnaire dataframe are disordered'

    # fit to & transform initial sample
    init_fa_min, init_f_labels, f_summary = run_fa(init_ques_df_min, init_corrmat_min, n_comps, rotation)
    init_df[[f'{f}_thresh{loading_thresh}' for f in init_f_labels]] = init_fa_min.X_reduced
    
    # pickle the fa dictionary
    fa_dict = init_fa_min.output()
    fa_dict['factor_labels'] = f_summary
    pickle_file(fa_dict, f'{base_dir}/factor_analyses/fa_{rotation}_{loading_thresh}.pkl')

    # transform validation sample 
    val_ques_df_min, val_corrmat_min = val_ques_df.loc[:, min_items], val_corrmat.loc[min_items, min_items]
    assert np.all(val_corrmat_min.columns == val_ques_df_min.columns), \
        'validation reduced corr. matrix & questionnaire dataframe are disordered'
    val_df[[f'{l}_thresh{loading_thresh}' for l in init_f_labels]] = init_fa_min.transform(val_ques_df_min)

    # #------------------------------------------------------
    # # are reduced FA loadings stable across samples?
    # #------------------------------------------------------

    # # test if fator scores are consistent within initial sample across full and reduced factor analyses
    # print('\nTesting consistency of factor scores within Initial sample:')
    # for f in f_labels:
    #     r, p = pearsonr(init_df[f], init_df[f'{f}_thresh{loading_thresh}'])
    #     print(f'  - {f}: r={r:.3f}, p={p:.3f}')

    # print('\nChecking reduced FA in Validation sample\n')

    # # fully fit to & transform validation sample
    # val_fa_min, val_f_labels, factor_summary = run_fa(val_ques_df_min, val_corrmat_min, n_comps, rotation)
    # val_df[[f'{l}_thresh{loading_thresh}_ind' for l in val_f_labels]] = val_fa_min.X_reduced

    # # correlate loadings across samples
    # assert np.all(init_fa_min.features == val_fa_min.features)
    # print('\nCorrelations between factor loadings across samples:')
    # for f in init_f_labels:
    #     init_f_loadings  = init_fa_min.loadings[:, np.where(np.array(init_f_labels) == f)[0][0]]
    #     val_f_loadings   = val_fa_min.loadings[:, np.where(np.array(val_f_labels) == f)[0][0]]
    #     cross_r, cross_p = pearsonr(init_f_loadings, val_f_loadings)
    #     print(f'  - {f}: r={cross_r:.3f}, p={cross_p:.3f}')

    #------------------------------------------------------
    # plot stuff
    #------------------------------------------------------

    # if plot: 
    #     # plot overall EFA
    #     corrmat = efa.plot_corrmat(figsize=(6,6))
    #     efa.plot_loadings(figsize=(10,3))
    #     efa.plot_components()
    #     # plot reduced FAs
    #     efa_min.plot_loadings(figsize=(12,3))
    #     repl_efa_min.plot_loadings(figsize=(12,3))
        
    #     fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
    #     ax = axs[0]
    #     ax.set_title('Initial sample scores')
    #     sns.histplot(init_df[factor_labels[0]], color='blue', alpha=0.5, stat='density', bins=20, ax=ax)
    #     sns.histplot(init_df[factor_labels[1]], color='red', alpha=0.5, stat='density', bins=20, ax=ax)
    #     sns.histplot(init_df[factor_labels[2]], color='green', alpha=0.5, stat='density', bins=20, ax=ax)
    #     ax = axs[1]
    #     ax.set_title('Validation sample scores')
    #     sns.histplot(repl_df[factor_labels[0]], color='blue', alpha=0.5, stat='density', bins=20, ax=ax)
    #     sns.histplot(repl_df[factor_labels[1]], color='red', alpha=0.5, stat='density', bins=20, ax=ax)
    #     sns.histplot(repl_df[factor_labels[2]], color='green', alpha=0.5, stat='density', bins=20, ax=ax)

    #     plt.show()

print('\n\nDone!')

#------------------------------------------------------
# save output
#------------------------------------------------------

data = pd.concat([init_df, val_df], axis=0)
for f in init_f_labels: # drop any rows that have a nan
    data = data.dropna(subset=[f])
data.reset_index(drop=True, inplace=True)
print(f'n={len(data)}')
data.to_excel(f'{base_dir}/data/All-data_summary_n{len(data)}.xlsx', index=False)
print('Saved')