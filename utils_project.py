import sys, os, glob, random, warnings, itertools
from six.moves import cPickle as pickle
import patsy
import numpy as np
from numpy.linalg import norm
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA, PLSRegression, PLSCanonical
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import make_pipeline
import scipy
from scipy.stats import (
    ttest_1samp, kendalltau, chi2_contingency,
    wilcoxon, pearsonr, spearmanr, gaussian_kde, zscore
)
from scipy.spatial import distance
import statsmodels.api as sm
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.regression import linear_model
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import shapely
from collections import Counter

warnings.filterwarnings("ignore") 
pd.set_option('display.precision', 4)
from factor_analysis import FactorAnalysis


#---------------------------------------------------------------------------------------------------------------------------------------------
# data 
#---------------------------------------------------------------------------------------------------------------------------------------------


user = os.path.expanduser('~')
if user == '/Users/matty_gee':
    base_dir = f'{user}/Desktop/projects/SNT-online_behavioral'
fig_dir = './figs'

# load data if exists
def update_sample_dict(data):
    return {'Initial': {'data': data[data['sample']==0].reset_index(drop=True), 'color': 'darkorchid'}, 
            'Validation': {'data': data[data['sample']==1].reset_index(drop=True), 'color': 'royalblue'},
            'Combined': {'data': data, 'color': 'black'}}

try: 
    # clean up a bit
    data = pd.read_excel(glob.glob(f'{base_dir}/Data/All-data_summary_n*.xlsx')[0]) # new factors included
    data.drop_duplicates(subset=['sub_id'], keep='first', inplace=True)
    # data = data[~data['factor_social_quartimax_thresh25'].isna()].reset_index(drop=True)
    data['demo_gender_1W'] = (data['demo_gender_1W'] == 1) * 1

    # add in the task version as a one-hot
    encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', drop='first')
    onehot = encoder.fit_transform(data['task_ver'].values.reshape(-1, 1)).toarray()
    onehot_cols = list(encoder.categories_[0][1:])
    other_controls = onehot_cols
    for i, colname in enumerate(onehot_cols):
        data[colname] = onehot[:, i]

    # create a dictionary
    sample_dict   = update_sample_dict(data)
    samples       = list(sample_dict.keys())
    sample_colors = [sample_dict[sample]['color'] for sample in sample_dict]
    for sample in sample_dict:
        print(f"{sample} n={len(sample_dict[sample]['data'])}")
except Exception as e:
    print(f'Could not load data: {e}')
    
try: 
    questionnaire_items = pd.read_excel(f'{base_dir}/questionnaire_items.xlsx')
    questionnaire_items = questionnaire_items.drop_duplicates('item')
    print('Questionnaire items loaded')
except Exception as e:
    print(f'Could not load questionnaire items: {e}')


#---------------------------------------------------------------------------------------------------------------------------------------------
# columns, labels etc
#---------------------------------------------------------------------------------------------------------------------------------------------

# data = flip_power(data) # gotta come up with a better way to do this...
demo_controls  = ['demo_age', 'demo_gender_1W', 'demo_race_white', 'iq_score', 'disorder'] #  'after_vaccine_fda'
all_controls = demo_controls + other_controls + ['memory_mean']
sample_colors = ['darkorchid', 'royalblue']

# for factor analysis & pls analysis
fa_prefixes   = ['oci', 'zbpd', 'sds', 'aes', 'sss', 'lsas_av', 'apdis', 'bapq']
ques_labels   = ['OCD', 'Borderline', 'Depression', 'Apathy', 'Schizotypy', 'Social Anxiety', 'Social Avoidance', 'Autism'] # match prefixes
all_prefixes  = fa_prefixes + ['audit', 'aq', 'eat', 'pid5', 'pdi', 'stai_t', 'stai_s', 'pq16', 'pss', 'ucls', 'dtm', 'dtn', 'dtp', 'ucls', 'sh']

ques_labels = {
    'oci': 'Obsessive Compulsive',
    'sds': 'Depression',
    'aes': 'Apathy',
    'sss': 'Schizotypy',
    'lsas': 'Social Anxiety',
    'lsas_av': 'Social Anxiety',
    'apdis': 'Avoidant Personality',
    'zbpd': 'Borderline Personality',
    'bapq': 'Autism',
}

# which fa to test more
social_factor     = 'factor_social_quartimax_thresh25'
mood_factor       = 'factor_mood_quartimax_thresh25'
compulsive_factor = 'factor_compulsive_quartimax_thresh25'
factor_labels = [social_factor, mood_factor, compulsive_factor]
factor_names  = ['Social Avoidance', 'Mood', 'Compulsion']
factor_dict   = dict(zip(factor_labels, factor_names))

behav_labels = {
    'affil_mean_mean':['Affiliation', "#4374B3"],
    'affil_mean_mean_z':['Affiliation', "#4374B3"],
    'affil_mean_mean_adj':['Affiliation', "#4374B3"],
    'affil_centroid_mean':['Affiliation', "#4374B3"],
    'power_mean_mean':['Power', "#FF0B04"],
    'power_mean_mean_z':['Power', "#FF0B04"],
    'power_mean_mean_adj':['Power', "#FF0B04"],
    'power_centroid_mean':['Power', "#FF0B04"],
    'positive_mean_mean':['Control', '#3CA926'],
    'positive_mean_mean2':['Control', '#3CA926'],
    'affil_mean_mean * power_mean_mean':['Affil*Power', '#6a329f'],
    'pov_2d_dist_mean':['POV distance', '#1ACBC0'],
    'pov_2d_dist_mean_mean':['POV distance', '#1ACBC0'],
    'pov_2d_angle_mean_cos':['POV angle', '#1ACBC0'],
    'pov_2d_angle_mean_mean_cos': ['POV angle', '#1ACBC0'],
    'pov_2d_angle_mean_mean_sin': ['POV angle', '#1ACBC0'],
    'neu_2d_dist_mean':['Neu distance', '#20CE5C'],
    'neu_2d_dist_mean_mean':['Neu distance', '#20CE5C'],
}

family        = ['marriage', 'dating', 'children', 'parents', 'relatives', 'inlaws']
nonfamily     = ['friends', 'school', 'neighbors', 'workNonsupervision', 'workSupervision', 'religion', 'volunteer', 'extraGroup1']
relationships = family + nonfamily
try: 
    relationship_cols = np.unique([('_').join(c.split('_')[0:2]) for c in data.columns if c.split('_')[0] in relationships])
    nonfamily_cols = [c for c in relationship_cols if c.split('_')[0] in nonfamily]
    family_cols    = [c for c in relationship_cols if c.split('_')[0] in family]
except:
    print('no data to get the relationhip columns from')


#---------------------------------------------------------------------------------------------------------------------------------------------
# task stuff
#---------------------------------------------------------------------------------------------------------------------------------------------


task = pd.read_excel('snt_details.xlsx')
task.sort_values(by='cogent_onset', inplace=True)
decision_trials = task[task['trial_type'] == 'Decision']
dtype_dict = {'decision_num': int,
                'scene_num': int,
                'char_role_num': int,
                'char_decision_num': int,
                'cogent_onset': float}
decision_trials = decision_trials.astype(dtype_dict) # ensure correct dtypes
decision_trials.reset_index(inplace=True, drop=True)
character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss', 'neutral'] # in order of role num in snt_details

def get_coords(df, which='task', include_neutral=False):

    # if its a series, turn into dataframe
    if isinstance(df, pd.Series): df = df.to_frame().T
    
    if include_neutral: roles = character_roles
    else: roles = [role for role in character_roles if role != 'neutral']
    relationship_cols = np.unique([('_').join(c.split('_')[0:2]) for c in df.columns if c.split('_')[0] in relationships])
    nonfamily_cols = [c for c in relationship_cols if c.split('_')[0] in nonfamily]
    family_cols    = [c for c in relationship_cols if c.split('_')[0] in family]

    if which == 'task':
        return reshape_dataframe(df, [[f'affil_mean_{role}', f'power_mean_{role}'] for role in roles])
    elif which == 'dots':
        return reshape_dataframe(df, [[f'{role}_dots_affil', f'{role}_dots_power'] for role in roles])
    elif which == 'ratings':
        return reshape_dataframe(df, [[f'{role}_likability', f'{role}_impact'] for role in roles])
    elif which == 'family_dots':
        return reshape_dataframe(df, [[f'{role}_dots_affil_relationship', f'{role}_dots_power_relationship'] for role in family_cols])
    elif which == 'nonfamily_dots':
        return reshape_dataframe(df, [[f'{role}_dots_affil_relationship', f'{role}_dots_power_relationship'] for role in nonfamily_cols])
    elif which == 'family_ratings':
        return reshape_dataframe(df, [[f'{role}_likability_relationship', f'{role}_impact_relationship'] for role in family_cols])
    elif which == 'nonfamily_ratings':
        return reshape_dataframe(df, [[f'{role}_likability_relationship', f'{role}_impact_relationship'] for role in nonfamily_cols])
    elif which == 'relationship_dots': 
        return reshape_dataframe(df, [[f'{role}_dots_affil_relationship', f'{role}_dots_power_relationship'] for role in relationship_cols])
    elif which == 'relationship_ratings':
        return reshape_dataframe(df, [[f'{role}_likability_relationship', f'{role}_impact_relationship'] for role in relationship_cols])

def load_behav(sub_id, neutrals=True):
    behav_files = glob.glob(f'{base_dir}/Data/*/SNT/Behavior/*.xlsx')
    try:
        behav_fname = [f for f in behav_files if f'SNT_{sub_id}' in f][0]
        df = pd.read_excel(behav_fname)
    except:
        print(f'No behavior file for {sub_id}')
        return None
    if not neutrals:
        df = df[df['char_role_num'] != 6].reset_index(drop=True)
    return df


#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------


def isfinite(x):
    mask = np.isfinite(x)
    return x[mask], mask

def reshape_dataframe(df, cols):
    """
    Takes a DataFrame and a list of lists of column names.
    Returns a 3D array with shape (p, n, m), where p is the number of rows in the DataFrame,
    n is the number of lists, and m is the length of each list.
    """
    if not all(isinstance(item, list) for item in cols):
        raise ValueError("All items in cols must be lists.")
    array_3d = []
    for col_list in cols:
        selected_cols = df[col_list]
        array_3d.append(selected_cols.values)
    return np.array(array_3d).transpose(1, 0, 2)

def get_item(item):
    return questionnaire_items[questionnaire_items['item']==item]

flatten_lists = lambda l: [item for sublist in l for item in sublist]

def subset_df(df, ques_prefixes):
    ques_dfs = [df.filter(regex=(f"{ques}_.*")) for ques in ques_prefixes]
    ques_df = pd.concat(ques_dfs, axis=1)
    ques_df = ques_df[[c for c in ques_df if ('_att' not in c) & ('score' not in c)]]
    ques_df.insert(0, 'sub_id', df['sub_id'])
    return ques_df

def list_cols(substr):
    # list columns that contain a substring
    return list(data.filter(regex=substr).columns)

def rescale(data, center=50):
    """
    Rescale data to be between [-1, +]
    """
    return (data - center) / center

def get_cols(substr):
    # use wildcards to find columns in a dataframe
    return data.filter(regex=substr)
 
def get_rdv(x, metric='euclidean'):
    return symm_mat_to_ut_vec(pairwise_distances(x, metric=metric))

def symm_mat_to_ut_vec(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    if isinstance(mat, pd.DataFrame):
        mat = mat.values
    return mat[np.triu_indices(len(mat), k=1)]


#---------------------------------------------------------------------------------------------------------------------------------------------
# replication helpers
#---------------------------------------------------------------------------------------------------------------------------------------------


def flip_power(data):
    # find all columns w/ power in name and flip sign
    data_ = data.copy()
    for col in [col for col in data_.columns if ('power_' in col) & ('dots' not in col)]: 
        data_[col] = data_[col] * -1
    return data_

# plotting ols
def plot_regplot_replication(xvars, yvars, 
                             size=4, 
                             p_tail='p', ps=None, p_plot='right'):

    if len(xvars) < len(yvars): xvars *= len(yvars)
    if len(yvars) < len(xvars): yvars *= len(xvars)

    if len(yvars) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(size, size))
        axs = [axs]
    else:
        fig, axs = plt.subplots(1, len(yvars), figsize=(size*len(yvars), size))
    for i, (xvar, yvar) in enumerate(zip(xvars, yvars)):
        for s, sample in enumerate(samples[:2]):

            df    = sample_dict[sample]['data']
            color = sample_dict[sample]['color']

            ax = plot_regplot(df[xvar].values, df[yvar].values, ax=axs[i], color=color)
            ax.set_xlabel(xvar, fontsize=label_fontsize)
            ax.set_ylabel(yvar, fontsize=label_fontsize)

            # run ols and get pvalue (optional)
            if p_tail is not None:
                if ps is None:
                    ols_df = run_ols(X=xvar, y=yvar, data=df)[0]
                    p = ols_df[ols_df['x'] == xvar][p_tail].values[0]
                else:
                    p = ps[i, s] # plot, sample
                dx, dy = 0.025, 0.03
                if p_plot == 'right_top': 
                    plot_significance(ax, p, sig_level=4, color=color, x=0.95, y=0.98-s*dy, dx=-dx, fontsize=17)
                elif p_plot == 'left_top':
                    plot_significance(ax, p, sig_level=4, color=color, x=0.0, y=0.98-s*dy, dx=dx, fontsize=17)
                elif p_plot == 'right_bottom':
                    plot_significance(ax, p, sig_level=4, color=color, x=0.95, y=0.02+s*dy, dx=-dx, fontsize=17)
                elif p_plot == 'left_bottom':
                    plot_significance(ax, p, sig_level=4, color=color, x=0.0, y=0.02+s*dy, dx=dx, fontsize=17)
    plt.tight_layout()
    return fig, axs

def run_replicated_ols(Xs, ys, controls=demo_controls, data=None, 
                       alpha=0.05, which_p='p', filter_repl=True):
    ols_res = []
    if isinstance(Xs, str): Xs = [Xs]
    if isinstance(ys, str): ys = [ys]
    for X, y in itertools.product(Xs, ys):
        if isinstance(X, str): X = [X] # easier filtering
        for s, sample in enumerate(samples[:2]):
            if data is None: 
                df = sample_dict[sample]['data']
            else: 
                df = data[data['sample']==s]
            if controls is not None: # filter out if there is overlap between controls and X + y
                ols_df = run_ols(X, y, data=df, covariates=[c for c in controls if c not in X + [y]])[0]
            else:
                ols_df = run_ols(X, y, data=df)[0]
            ols_df.insert(0, 'sample', sample)
            ols_df = ols_df[ols_df['x'].isin(X)]
            ols_res.append(ols_df)
    ols_res = pd.concat(ols_res)

    if alpha is not None:
        ols_res['sig'] = ols_res[which_p] < alpha
        if filter_repl: 
            ols_res['sig'] = ols_res.groupby(['x', 'y'])['sig'].transform('sum')
            ols_res = ols_res.sort_values(by=['x', 'y', 'sig'], ascending=False)
            # only include if 2 in sig
            ols_res = ols_res[ols_res['sig'] == 2]

    return ols_res

def run_and_plot_replication_regplot(xvars, yvars, controls=None,
                                     alpha=0.05, which_p='p', filter_repl=True,
                                     size=4, ps=None, p_plot='right'):

    # run regression in each sample
    ols_repl = run_replicated_ols(xvars, yvars, controls=controls, 
                                  alpha=alpha, which_p=which_p, filter_repl=filter_repl)
    # plot what comes out
    if len(ols_repl) > 0:
        yvars, xvars = [], []
        for y, x in ols_repl[['y', 'x']].values:
            if y not in yvars: yvars.append(y)
            if x not in xvars: xvars.append(x)
        n_plots = np.max([len(yvars), len(xvars)])
        fig, axs = plot_regplot_replication(xvars, yvars, 
                                            ps=ols_repl['p'].values.reshape(n_plots, 2), 
                                            size=4,
                                            p_tail='p',
                                            p_plot='right_top')
        return fig, axs, ols_repl 
    else:
        print('No significant replications')


#---------------------------------------------------------------------------------------------------------------------------------------------
# plotting functions
#---------------------------------------------------------------------------------------------------------------------------------------------


alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] # for stars
tick_fontsize, label_fontsize, title_fontsize = 10, 13, 15
legend_title_fontsize, legend_label_fontsize = 12, 10
suptitle_fontsize = title_fontsize * 1.5
ec, lw = 'black', 1
bw = 0.15 
figsize, facet_figsize = (5, 5), (5, 7.5)

def plot_significance(ax, pvalue, sig_level=4, color=None, x=0.0, y=0.98, dx=0.015, fontsize=17):
    alphas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    if color is None: color = 'k'
    for i, alpha in enumerate(alphas[:sig_level]):
        if pvalue < alpha:
            ax.text(x+dx*i, y, "*", color=color, fontsize=fontsize, 
                    ha="left", va="top", transform=ax.transAxes)  

def add_sig_stars(x, y, df, demo_controls, ax, color):
    _, ols_obj = run_ols([x], y, df, covariates=demo_controls, n_splits=None, plot=False)
    x_change = 0.04    
    sigs = [0.05,0.01,0.005,0.001,0.0005,0.0001]
    for s, sig in enumerate(sigs):
        if ols_obj.pvalues[x] < sig:
            ax.text(0.225-x_change*s, 0.98, "*", color=color, fontsize=15, ha="left", va="top", transform=ax.transAxes)  

def plot_space_density(x, y, figsize=(5,5), ax=None, regression=False):
    
    '''
    '''

    # calculate the point density
    xy = np.vstack([x, y])
    z  = gaussian_kde(xy)(xy) 

    # sort so that the bright spots are on top
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # plot
    fig, axs = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'hspace': 0, 'wspace': 0,
                                                                'width_ratios': [5, 1], 'height_ratios': [1, 5]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")

    axs[1,0].set_ylim([-1,1])
    axs[1,0].set_yticks([-1,-.5,0,.5,1])
    axs[1,0].set_xlim([-1,1])
    axs[1,0].set_xticks([-1,-.5,0,.5,1])
    axs[1,0].axhline(y=0, color='black', linestyle='-', zorder=-1)
    axs[1,0].axvline(x=0, color='black', linestyle='-', zorder=-1)
    axs[1,0].set_xlabel('Affiliation')
    axs[1,0].set_ylabel('Power')

    # plot distributions on the sides
    sns.distplot(x, bins=20, ax=axs[0,0], color="Black")
    sns.distplot(y, bins=20, ax=axs[1,1], color="Black", vertical=True)

    # plot density
    axs[1,0].scatter(x, y, c=z, s=100)

    # plot regression
    if regression: sns.regplot(x, y, scatter=False, color='black', ax=axs[1,0])
    
    plt.show()

def random_colors(num_colors=10):
    from random import randint
    return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(num_colors)]

def plot_ols_models_metrics(results_df, metric='train_BIC', colors=None):
    
    if colors is None: colors = []
    # assign colors
    models = np.unique(results_df['model'])
    n_models = len(models)
    n_folds = list(results_df['model']).count(models[0])
    results_df['color'] = np.repeat(np.arange(n_models), n_folds, 0)

    if len(colors) == 0:
        colors = random_colors(num_colors=n_models)
        # color = iter(plt.cm.cool(np.linspace(0, 1, n_models))) # need more different colors
        # colors = []
        # for i in range(n_models): colors.append(list(next(color)))

    metric_df = results_df[results_df['metric'] == metric]
    mean_df = metric_df[metric_df['fold'] == 'mean']

    if 'IC' in metric: 
        ascending = True
    else:
        ascending = False
    mean_df.sort_values('value', ascending=ascending, inplace=True)
    ordered_models = mean_df['model'].values
    ordered_colors = mean_df['color'].values

    fig, ax = plt.subplots(figsize=(15,7))
    ax = sns.barplot(x='model', y='value', hue='fold', data=metric_df, order=ordered_models)
    for i, bar in enumerate(ax.patches):
        c = i % n_models
        bar.set_color(colors[ordered_colors[c]])
        bar.set_edgecolor(".2")
    ax.set_ylabel(metric, fontsize=20)
    ax.set_ylim(np.min(metric_df['value'])-np.min(metric_df['value'])*.05, 
                np.max(metric_df['value'])+np.max(metric_df['value'])*.05)
    ax.set_xticklabels([x for x in ordered_models], rotation=60, ha='right')
    plt.legend([],[], frameon=False)
    plt.show()

def create_subplots(grid_size, irregular_axes=None, figsize=(10,10), annotate=False):
    fig = plt.figure(figsize=figsize)

    # initialize a grid to keep track of occupied cells
    grid_occupancy = [[False]*grid_size[1] for _ in range(grid_size[0])]

    axs = []

    # make any irregular axes
    if irregular_axes is not None:
        for idx, shape in irregular_axes.items():
            axs.append(plt.subplot2grid(grid_size, idx, colspan=shape[1], rowspan=shape[0]))
            # Mark the cells as occupied
            for i, j in itertools.product(range(shape[0]), range(shape[1])):
                if idx[0]+i < grid_size[0] and idx[1]+j < grid_size[1]:
                    grid_occupancy[idx[0]+i][idx[1]+j] = True 

    # create 1x1 subplots in the remaining cells
    for i, j in itertools.product(range(grid_size[0]), range(grid_size[1])):
        if not grid_occupancy[i][j]:
            axs.append(plt.subplot2grid(grid_size, (i, j)))

    # add text to each ax 
    if annotate: 
        for i, ax in enumerate(axs):
            ax.text(0.5, 0.5, f'axs[{i}]', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    return fig, axs


#---------------------------------------------------------------------------------------------------------------------------------------------
# EFA
#---------------------------------------------------------------------------------------------------------------------------------------------


def name_factors(loadings, items, n_items=40):

    ques      = np.unique([ii.split('_')[0] for ii in items])
    loadings  = pd.DataFrame(loadings, index=items)
    factor_qs = {'mood':['sds', 'aes'], 'compulsive':['oci', 'zbpd'], 'social':['lsas', 'apdis', 'bapq']}

    loadings   = pd.DataFrame(loadings, index=items)
    summary_df = pd.DataFrame(columns=['factor_num', 'factor_label'] + 
                                       flatten_lists([[f'{q}_mean', f'{q}_sd']  for q in ques]) + 
                                       [f'item_{i+1}' for i in range(len(items))])
    for i in range(3):

        top_items = loadings.iloc[:,i].sort_values(ascending=False)

        # which questions load most on this factor
        most_common = Counter([ii.split('_')[0] for ii in top_items[:n_items].index]).most_common(10)
        most_common = [c[0] for c in most_common]
        
        # find which factors those questions come from
        factors = flatten_lists([[k for k, v in factor_qs.items() if mc in v] for mc in most_common])
        summary_df.loc[i, ['factor_num', 'factor_label']] = [i+1, factors[0]]

        # output the top items, in order
        summary_df.loc[i, [f'item_{j+1}' for j in range(len(items))]] = list(top_items.index)

        # get mean and sd for each questionnaire
        for q in ques: 
            mask  = [q in ii for ii in loadings.index]
            m, sd = np.mean(loadings[mask].iloc[:,i]), np.std(loadings[mask].iloc[:,i])
            summary_df.loc[i, [f'{q}_mean', f'{q}_sd']] = [m, sd]

    # return a warning if there are any non-unique labels
    if len(np.unique(summary_df['factor_label'])) < 3:
        print('WARNING: not unique labels')

    return summary_df

def run_fa(ques_df, corrmat, n_comps, rotation):

    # run factor analysis
    efa = FactorAnalysis()
    efa.fit_transform(ques_df, corr_matrix=corrmat, num_comps=n_comps, rotation=rotation)
    assert efa.X.shape[0] == ques_df.shape[0], 'reduced X matrix has wrong number of rows'

    # assign factor names
    factor_summary = name_factors(efa.loadings, efa.features)
    factor_labels = [f'factor_{l}_{rotation}' for l in factor_summary['factor_label'].values]
    
    return efa, factor_labels, factor_summary


#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------


def get_quadrant(x, y, origin=(0, 0)):
    """
        Returns the 2D quadrant (1-4) for a given point (x, y) with respect to an origin.
        Returns NaN if either coordinate is NaN.
        Returns 0 if the point lies on one of the axes.
    """
    assert isinstance(x, (int, float, np.number)), "x must be a number"
    assert isinstance(y, (int, float, np.number)), "y must be a number"
    assert isinstance(origin, tuple) and len(origin) == 2, "origin must be a tuple with two elements"
    assert all(isinstance(i, (int, float, np.number)) for i in origin), "both elements of origin must be numbers"

    # Handle NaN values
    if np.isnan(x) or np.isnan(y):
        return np.nan

    # Determine quadrant
    if x > origin[0] and y > origin[1]:
        return 1
    elif x < origin[0] and y > origin[1]:
        return 2
    elif x < origin[0] and y < origin[1]:
        return 3
    elif x > origin[0] and y < origin[1]:
        return 4
    else:
        return 0  # On one of the axes
    
def get_quadrant_vertices(origin, size):
    """
        Define the vertices of the four quadrants centered around a given origin and with given size.
        
        Arguments:
        - origin: Tuple of x, y coordinates specifying the center of the quadrants.
        - size: The size of each dimension, extending equally in the positive and negative directions from the origin.
        
        Returns:
        - A 3D numpy array where each slice is a 2D array of vertices defining a quadrant.
    """
    assert isinstance(origin, tuple) and len(origin) == 2, "origin must be a tuple with two elements"
    assert all(isinstance(i, (int, float, np.number)) for i in origin), "both elements of origin must be numbers"
    assert isinstance(size, (int, float, np.number)), "size must be a number"

    x, y = origin
    return np.array([
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],  # Quadrant 1 vertices
        [[x - size, y], [x, y], [x, y + size], [x - size, y + size]],  # Quadrant 2 vertices
        [[x, y], [x - size, y], [x - size, y - size], [x, y - size]],  # Quadrant 3 vertices
        [[x + size, y], [x, y], [x, y - size], [x + size, y - size]]   # Quadrant 4 vertices
    ])

def calc_quadrant_overlap(coords, origin=(0, 0), size=1, float_dtype="float32", verbose=False):
    """
        Calculate the proportion of a polygon's area that overlaps with each of four quadrants.
        The polygon is defined by the 2D coordinates `coords`.
        The origin is defined by the 2D coordinates `origin`.
    """
    assert isinstance(coords, np.ndarray), "coords must be a numpy array"
    assert coords.shape[1] == 2, "coords must be 2D coordinates"
    assert isinstance(origin, tuple) and len(origin) == 2, "origin must be a tuple with two elements"
    assert all(isinstance(i, (int, float, np.number)) for i in origin), "both elements of origin must be numbers"
    assert isinstance(size, (int, float, np.number)), "size must be a number"
    assert isinstance(float_dtype, str), "float_dtype must be a string"

    quad_vertices = get_quadrant_vertices(origin, size)

    # Filter out rows in coords with NaNs
    coords = coords[~np.isnan(coords).any(axis=1)]

    try: 
        convexhull = scipy.spatial.ConvexHull(coords)
        polygon = shapely.geometry.Polygon(coords[convexhull.vertices])
        overlap = [polygon.intersection(shapely.geometry.Polygon(v)).area / polygon.area for v in quad_vertices]
        return np.array(overlap, dtype=float_dtype)
    except Exception as e:
        # Optionally, you can print the exception for debugging
        if verbose: print(f"Exception encountered: {e}")
        return np.full(4, np.nan, dtype=float_dtype)
    
def calc_quarant_relative(quad_scores, quad_means=None):
    # takes into an array of 4 columns (quadrants) and n rows (participants)
    # calculates the relative preference for each quadrant (how much more than expected do participants overlap in each quadrant)

    # adjust by sample mean to get how much more than expected do participants overlap in each quadrant
    if quad_means is None: quad_means = np.nanmean(quad_scores, axis=0)
    quad_relative = quad_scores - quad_means

    # assign rows values depending on which quadrant they have largest value in, but if nans then give nan
    quad_top = np.where(np.isnan(quad_scores).all(axis=1), np.nan, np.argmax(quad_scores, 1) + 1)

    # relabel quadrants
    # relabel = {1: 'High Affil & Low Power', 2: 'Low Affil & Low Power', 3: 'Low Affil & High Power', 4: 'High Affil & High Power'}
    quad_relative = np.hstack([quad_relative, quad_top[:, np.newaxis]])
    quad_relative = pd.DataFrame(quad_relative, 
                                 columns=['quadrant1_tendency_relative', 'quadrant2_tendency_relative', 
                                          'quadrant3_tendency_relative', 'quadrant4_tendency_relative', 'quadrants_tendency_relative'])
    return quad_relative

def calc_area(xy):
    try: 
        return scipy.spatial.ConvexHull(xy).volume
        # return scipy.spatial.ConvexHull(xy[~np.isnan(xy)].reshape(-1, 2)).volume
    except:
        return np.nan



#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

def calc_perm_pvalue(stat, perm_stats, alternative='two-sided'):
    '''
    Calculate p-value of test statistic compared to permutation generated distribution.
    Finds percentage of permutation distribution that is at least as extreme as the test statistic

    Arguments
    ---------
    stat : float
        Test statistic
    perm_stats : array of floats
        Permutation generated (null) statistics
    alternative : str (optional)

    Returns
    -------
    float 
        p-value

    [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    n = len(perm_stats)
    p_greater = (np.sum(perm_stats >= stat) + 1) / (n + 1) # add one for a continuity correction
    p_less    = (np.sum(perm_stats <= stat) + 1) / (n + 1)

    if alternative == 'two-sided':   
        # accounts for scenario where the observed statistic is exactly at the median of the permutation distribution
        return 2 * min(p_greater, p_less) if p_greater + p_less > 1 else 2 * max(p_greater, p_less) 
    elif alternative == 'greater':  
        return p_greater
    elif alternative == 'less':     
        return p_less
    
def plot_barplot(x, y, ax=None, color=None, pal=None, **kwargs):
    if ax is None: ax = plt.gca()
    sns.barplot(x=x, y=y, color=color, palette=pal, 
                edgecolor='black', 
                ax=ax, **kwargs)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    return ax

def run_ols(X, y=None, covariates=None, data=None,
            loglik_ratio=False, 
            popmean=0, 
            verbose=False):
    '''
        Run ordinary least squares regression or 1-sample t-test
        Wrapper for statsmodels OLS
        Cleans up strings to be compatible with patsy & statsmodel formulas

        Inputs:
            X: Dataframe, series, list strings, string, or array
            y (optional): Dataframe, series, list strings, string, or array
            data (optional): Dataframe, for use if X & y are strings
            covariates (optional): list of strings, for use if X & y are strings

            popmean (optional): float or int, for use if t-test

            plot (optional): bool, plot the predicted vs observed values
        
        Outputs:
            out_df: Dataframe, summary of regression results
            ols_obj: statsmodels OLS object
    '''

    # helpers
    def clean_string(string):
        if '.' in string:
            string = string.replace('.', '_')
        return string

    def make_dataframe(a):
        if isinstance(a, pd.DataFrame): 
            return a.reset_index(drop=True)
        if isinstance(a, pd.Series): 
            return a.to_frame().reset_index(drop=True)
        else:
            if isinstance(a, list): a = np.array(a)
            if a.ndim == 1: a = a[:, np.newaxis]
            return pd.DataFrame(a)

    def nanzscore(arr):
        arr_z = np.array([np.nan] * len(arr)) # create an array of nans the same size as arr
        arr_z[np.isfinite(arr)] = scipy.stats.zscore(arr[np.isfinite(arr)]) # calculate the zscore of the masked array
        return arr_z

    # handle strings
    if isinstance(X, str) or (isinstance(X, list) and isinstance(X[0], str)):

        # is it a formula?
        if '~' in X:
           
            df = data.copy()
            y, X = patsy.dmatrices(X, df, return_type="dataframe")

        # if not, construct the formula
        else:

            if (isinstance(X, str)): X = [X]
            cols = list(np.unique(flatten_lists([x.split('*') if '*' in x else [x] for x in X] + [[y]])))
            if covariates is not None: 
                if (isinstance(covariates, str)): covariates = [covariates]
                cols.extend(covariates)
            df = data[cols].copy() # restrict to the columns provided (exclude interaction labels)

            # remove '.' & rename columns
            X_labels = X.copy()
            if any(['.' in col for col in df.columns]):
                for i, x_label_ in enumerate(X):
                    X_labels[i] = clean_string(x_label_)
                    df.rename(columns={x_label_:X_labels[i]}, inplace=True)
                df.rename(columns={y:clean_string(y)}, inplace=True)

            # create dataframes from patsy formula
            X_labels = X_labels + covariates if covariates is not None else X_labels
            y, X = patsy.dmatrices(f'{clean_string(y)} ~ ' + ' + '.join(X_labels), df, return_type="dataframe")

    # handle series, dataframes, arrays or lists
    else: 

        # build the regression from arrays
        if y is not None: 
            y = make_dataframe(y)
            if isinstance(y.columns[0], int): y.columns = ['y']
            X = make_dataframe(X)
            if all([isinstance(c, int) for c in X.columns]): X.columns = [f'x{i+1}' for i in range(X.shape[1])]
            X = sm.add_constant(X)
            X.rename(columns={'const':'Intercept'}, inplace=True)

        # if y is None, then its a 1-sample t-test: X ~ intercept
        else:
            y = make_dataframe(X)
            xcol = y.columns[0] if not isinstance(y.columns[0], int) else 'x1'
            X = pd.DataFrame(np.ones(len(X)), columns=[xcol])
            y.rename(columns={y.columns[0]:f' against {popmean}'}, inplace=True) # rename so easier to interpret in output
            y = y - popmean

    # z-score continuous variables (ols only)
    y[y.columns[0]] = nanzscore(y[y.columns[0]])
    if verbose: print(f'Z-scored y: {y.columns[0]}')
    for col in X.columns:
        if (X.dtypes[col] in ['float64', 'int64']) & (col not in ['Intercept', 'const']) & (X[col].nunique() > 2):
            X[col] = nanzscore(X[col])
            if verbose: print(f'Z-scored x: {col}')
        
    # run the regression 
    try: 
        ols    = sm.OLS(y, X, missing='drop').fit()
        out_df = pd.DataFrame(ols.summary2().tables[1])

        if loglik_ratio:
            ll_0 = sm.OLS(y, sm.add_constant(np.ones(len(y))), missing='drop').fit() # null model (intercept-only)
            ll_r, ll_r_p = ols.compare_lr_test(ll_0)[:2] # compare against null
    
    except Exception as e:
        print(f'Error: {e}')
        return X, y

    # clean up the output
    out_df.columns = ['beta', 'se', 't', 'p', '95%_lb', '95%_ub']
    out_df.reset_index(inplace=True)
    out_df.rename(columns={'index': 'x'}, inplace=True)
    out_df['X'] = (' + ').join(X.columns.tolist())
    out_df['y'] = y.columns[0]
    out_df['dof'] = ols.df_resid
    # out_df['z'] = scipy.stats.norm.ppf(1 - out_df['p'] / 2) # z-score for 2-tailed test

    if loglik_ratio: 
        out_df['ll'] = ols.llf
        out_df['llr'] = ll_r
        out_df['llr_p'] = ll_r_p
    out_df['adj_rsq'] = np.round(ols.rsquared_adj, 3)
    out_df['bic'] = np.round(ols.bic, 2)
    out_df['aic'] = np.round(ols.aic, 2)
    out_df['p_right'] = [p/2 if b > 0 else 1-p/2 for b, p in zip(out_df['beta'].values, out_df['p'].values)]
    out_df['p_left']  = [p/2 if b < 0 else 1-p/2 for b, p in zip(out_df['beta'].values, out_df['p'].values)]

    if loglik_ratio:
        out_df = out_df[['X', 'y', 'x', 'dof', 'll', 'llr', 'llr_p', 'adj_rsq', 'bic', 'aic', 'beta', 'se', 
                         '95%_lb', '95%_ub', 't', 'p', 'p_left', 'p_right']]
    else:
        out_df = out_df[['X', 'y', 'x', 'dof', 'adj_rsq', 'bic', 'aic', 'beta', 'se', 
                         '95%_lb', '95%_ub', 't', 'p', 'p_left', 'p_right']]
    if 'against' in y.columns[0]: out_df['X'] = 't-test'

    return out_df, ols

def save_figure(fig, fname, formats=None):
    if isinstance(formats, str): 
        formats = [formats]
    elif formats is None: formats = ['png']
    formats = [format[1:] if format.startswith('.') else format for format in formats]
    if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.svg'):
        fname = fname[:-4]
    for format in formats:
        fig.savefig(f'{fname}.{format}',format=format, bbox_inches='tight', dpi=1200)

def parametric_ci(coeffs, conf=99):
    '''
        Computes parametric confidence interval: 
        Inputs:
            coeffs: 1-d array coefficients to compute confidence interval over
            conf: confidence level, in integers

        same as: scipy.stats.t.interval
    '''
    from scipy.stats import sem, t
    n  = len(coeffs)
    df = n - 1
    m  = np.mean(coeffs)
    se = sem(coeffs)
    h  = se * t.ppf((1 + (conf/100)) / 2, df)
    # TO DO - use normal dist, rather than t dist, if n >= 30
    return [m - h, m + h]

def zscore_masked(arr):
   arr_z = np.array([np.nan] * len(arr)) # create an array of nans the same size as arr
   mask = np.isfinite(arr) # mask the array
   arr_z[mask] = scipy.stats.zscore(arr[mask]) # calculate the zscore of the masked array
   return arr_z

def max_scale(data):
    return data/np.max(data)

def get_quadrant_levels(x, y, levels=None):
    if levels is None:
        levels = ['low', 'high']
    if (x == levels[1]) & (y == levels[1]):
        return 1
    elif (x == levels[0]) & (y == levels[1]):
        return 2
    elif (x == levels[0]) & (y == levels[0]):
        return 3
    elif (x == levels[1]) & (y == levels[0]):
        return 4
    elif np.isnan(x) | np.isnan(y):
        return np.nan

def adj_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1)

def factor_model_metrics(models, df, factor_label, covariates, n_splits):
    res = []
    for model in models:
        if n_splits is None:
            ols_df, _ = run_ols(model, factor_label, df, covariates=covariates, n_splits=n_splits, plot=False)
        else:
            ols_df = run_ols(model, factor_label, df, covariates=covariates, n_splits=n_splits, plot=False)
        df_ = pd.melt(ols_df.reset_index(), id_vars='index', value_vars=['train-BIC', 'train-r_squared_adj', 'test-BIC', 'test-r_squared_adj', 'test-RMSE'], 
                     var_name='metric', value_name='value')
        df_.rename(columns={'index':'fold'}, inplace=True)
        df_.insert(0, 'model', (' + ').join(model))
        res.append(df_)
    return pd.concat(res,0)

def plot_pca_2d(X):
    pca = PCA(n_components=2).fit(X)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i}",
            linewidth=5,
            color=f"C{i + 4}")
    plt.gca().set(
        aspect="equal",
        title="Principal components",
        xlabel="first feature",
        ylabel="second feature",
    )
    plt.legend()
    plt.show()
    
def optimise_pls_cv(X, y, n_comp, plot_components=True):
 
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE 
       Source: https://nirpyresearch.com/partial-least-squares-regression-python/
    '''
 
    
    n_targets = y.shape[1]
    
    # Cross-validated pls
    mse = []
    component = np.arange(1, n_comp)
    for i in component:
        pls = PLSRegression(n_components=i)
        y_cv = cross_val_predict(pls, X, y, cv=5) 
        mse.append(mean_squared_error(y, y_cv))
    
    #--------------------------------------------------
    # find minimum in MSE to find num of components
    #--------------------------------------------------
    
    msemin = np.argmin(mse)
    n_comps = msemin + 1   # optimal number of components 
    if plot_components is True:
        with plt.style.context(('ggplot')):
            fig = plt.figure(figsize=(3,3))
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=0)
            plt.show()
 
    # Calibration: entire dataset
    pls_cal = PLSRegression(n_components=n_comps)
    pls_cal.fit(X, y)
    X_tr_cal = pls_cal.transform(X) # transform X variables w/ the estimated weights that maximize covariance between X & Y
    y_preds_cal = pls_cal.predict(X)
 
    # Cross-validation
    pls_cv = PLSRegression(n_components=n_comps)
    y_preds_cv = cross_val_predict(pls_cv, X, y, cv=10)    
    
    # metrics
    score_c  = r2_score(y, y_preds_cal)
    score_cv = r2_score(y, y_preds_cv)
    mse_c  = mean_squared_error(y, y_preds_cal)
    mse_cv = mean_squared_error(y, y_preds_cv)
    
    print(f"Suggested number of components = {n_comps}")
    print(f'Calibration: R2 = {np.round(score_c,3)}, MSE = {np.round(mse_c,3)}')
    print(f'Cross-validation: R2 = {np.round(score_cv,3)}, MSE ={ np.round(mse_cv,3)}')

    #     # Plot regression and figures of merit
    #     rangey = max(y) - min(y)
    #     rangex = max(y_c) - min(y_c) 
    #     # Fit a line to the CV vs response
    #     z = np.polyfit(y, y_c, 1)
    #     with plt.style.context(('ggplot')):
    #         fig, ax = plt.subplots(figsize=(9, 5))
    #         ax.scatter(y_c, y, c='red', edgecolors='k')
    #         #Plot the best fit line
    #         ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
    #         #Plot the ideal 1:1 line
    #         ax.plot(y, y, color='green', linewidth=1)
    #         plt.title('$R^{2}$ (CV): '+str(score_cv))
    #         plt.xlabel('Predicted $^{\circ}$Brix')
    #         plt.ylabel('Measured $^{\circ}$Brix')
    #         plt.show()

    #--------------------------------------------------
    #--------------------------------------------------

    # if (n_targets > 1) or (n_comps > 1):
    #     if n_comps == 1: X_tr_cal = X_tr_cal[:,np.newaxis]
    #     fig, axs = plt.subplots(n_comps, n_targets, figsize=(3*n_comps, 3*n_targets))
    #     for r in range(n_targets):
    #         for c in range(n_comps):
    #             axs[c,r].scatter(X_tr_cal[:,c], y[:,r], alpha=0.3, label="ground truth")
    #             axs[c,r].scatter(X_tr_cal[:,c], y_preds_cal[:,r], alpha=0.3, label="predictions")
    #             axs[c,r].set(xlabel=f"Projected X {c+1}", ylabel=f"True y {r+1}", title=f"Component {c+1}")
    #             axs[c,r].legend()
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     fig, ax = plt.subplots(figsize=(3,3))
    #     ax.scatter(X_tr_cal, y, alpha=0.3, label="ground truth")
    #     ax.scatter(X_tr_cal, y_preds_cal, alpha=0.3, label="predictions")
    #     ax.set(xlabel=f"Projected X", ylabel=f"True y", title=f"Component")
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.show()
    
    return pls_cal