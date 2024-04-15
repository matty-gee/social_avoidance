from utils_project import *
from preprocess import ComputeBehavior2

character_roles_ = character_roles # for summary statistics across characters: incl. neutral or not?


#------------------------------------------------------------------------------------------------------------------------
# load & merge 
#------------------------------------------------------------------------------------------------------------------------


print('\nLOADING & MERGING DATA')
print('--------------------------------------------------')

dfs = []
for s, sample_dir in enumerate(['Initial_2021', 'Replication_2022']):

    summary_dir = f'{base_dir}/data/{sample_dir}/Summary/Individual_summaries'

    # if summary_fname := glob.glob(f'{summary_dir}/../All-data_summary_n*'):
    #     print(f'{sample_dir}: already processed')
    #     df = pd.read_excel(summary_fname[0])
    # else:

    # these should have happened already - TODO add the relevant code here
    print(f' -- {sample_dir}: merging data')
    beh_fname  = glob.glob(f'{summary_dir}/SNT-behavior_n*')[0]
    post_fname = glob.glob(f'{summary_dir}/SNT-posttask_n*')[0]
    ques_fname = glob.glob(f'{summary_dir}/Questionnaire_summary_n*')[0]
    df = merge_dfs([pd.read_excel(x) for x in [beh_fname, post_fname, ques_fname]])
    df.insert(1, 'sample', s)

    # NOT SURE WHAT TO DO RE: THIS...
    # df = flip_power(df) # hacky, but flips the orientation of the power dimension to make slightly more interpretable
    
    df.to_excel(f'{summary_dir}/../All-data_summary_n{len(df)}.xlsx', index=False) # save full df

    print(f' -- {sample_dir}: raw n={len(df)}')
    dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True)


#------------------------------------------------------------------------------------------------------------------------
# exclusions
#------------------------------------------------------------------------------------------------------------------------


print('\nEXCLUDING BAD SUBJECTS')
print('--------------------------------------------------')

# exclude if dont have behavior!
df = df[np.isfinite(df['affil_coord_first'])]
print(f' -- complete snt data: n={len(df)}')


# exclude improbable rts (+/- 2 SDs from mean)
sd = np.std(df['reaction_time_mean']) * 2
m  = np.mean(df['reaction_time_mean'])
lb, ub = m - sd, m + sd
df = df[(df['reaction_time_mean'] > lb) & (df['reaction_time_mean'] < ub)]
print(f' -- RT within 2 SDs: n={len(df)}')


# exclude memory below chance threshold
df = df[df['memory_mean'] > 1/6] # all 6 characters presented on each trial
print(f' -- memory above chance: n={len(df)}')


# exclude if dots didnt work
dots = df[[f'{role}_dots_affil' for role in character_roles]]
df = df[((np.sum(dots == -.92, 1) != 6) & np.isfinite(df['first_dots_affil']))]
print(f' -- dots worked: n={len(df)}')
# print(f' -- dots worked (incl. a flag): n={np.sum(df["dots_worked"])}')


# exclude if didnt complete questionnaires
df = df[df['bapq_score'] != 0].reset_index(drop=True) # 0 means they didnt have these
print(f' -- have BAPQ: n={len(df)}')
df = df[np.isfinite(df['lsas_av_score'])] # if didnt complete lsas
print(f' -- have LSAS: n={len(df)}')
att_checks = np.sum(df[['oci_att', 'aes_att', 'sds_att']] == [1, 0, 4], axis=1) == 3
df = df[att_checks]
print(f' -- attention checks: n={len(df)}')


# remove duplicate subjects
df = df.drop_duplicates(subset=['sub_id'], keep='first').reset_index(drop=True)
print(f' -- removed duplicates n={len(df)}')

## for running hetcor
# ques_df = subset_df(df, all_prefixes)
# ques_df.to_csv(f'{summary_df}/Questionnaire_items_n{len(ques_df)}.csv', index=False)


#------------------------------------------------------------------------------------------------------------------------
# other data
#------------------------------------------------------------------------------------------------------------------------


print('\nMERGING WITH OTHER DATA')
print('--------------------------------------------------')

# older factor analysis
if 'social_factor_fa01' not in df.columns:
    fa01 = pd.read_excel(f'{base_dir}/data/FA01_n912.xlsx') 
    fa01.drop_duplicates(subset=['sub_id'], keep='first', inplace=True)
    fa01.columns = ['sub_id', 'sample'] + [f'{c}_fa01' for c in fa01.iloc[:, 2:].columns]
    df = df.merge(fa01, on=['sub_id', 'sample'], how='left')
    df.reset_index(drop=True, inplace=True)

# social controllability data (not currently using these data but maybe becomes interesting by chance)
soc_control = pd.read_excel(f'{base_dir}/data/Initial_2021/Social_controllability_n1342.xlsx')
df = df.merge(soc_control, on='sub_id', how='left')
print('Merged')



print('\nADDING OTHER VARIABLES')
print('--------------------------------------------------')


#-------------------------------------------------------------------------------------
# self-ratings
#-------------------------------------------------------------------------------------


print('Adding self-rating related variables')

self_xy = rescale(df[['self_likability', 'self_impact']], center=50)
df['self_quadrant'] = [get_quadrant(xy[0], xy[1]) for xy in self_xy.values]

self_xy_relative = self_xy - np.nanmean(self_xy, axis=0)
df['self_quadrant_relative'] = [get_quadrant(xy[0], xy[1]) for xy in self_xy_relative.values]

df['likability_selfOther_avg'] = np.mean(df[['likability_mean', 'self_likability']], 1)
df['impact_selfOther_diff'] = df['impact_mean'].values - df['self_impact'].values
for col in ['likability_mean', 'self_likability', 'impact_mean', 'self_impact']:
    df[f'{col}_rescaled'] = (df[col] - 50) / 50


#--------------------------------------------------------------------------------------------------
# neutral character values
#--------------------------------------------------------------------------------------------------


print('Adding neutral character values')

df[['affil_coord_neutral', 'power_coord_neutral', 'affil_mean_neutral', 'power_mean_neutral']] = 0, 0, 0, 0
df[['neu_2d_dist_neutral', 'neu_2d_dist_mean_neutral', 'pov_2d_dist_neutral', 'pov_2d_dist_mean_neutral']] = 0, 0, 6, 6
df[['pov_2d_angle_neutral', 'neu_2d_angle_neutral']] = np.deg2rad(90), np.deg2rad(0)

# adjust the dots v& task variables using neutral dots location
for dim in ['affil', 'power']:

    # neutral dots coordinates
    adjust = df[f'neutral_dots_{dim}'].values[:,np.newaxis]

    # dots
    cols = [f'{r}_dots_{dim}' for r in character_roles_]
    df[[f'{c}_adj' for c in cols]] = df[cols].values - adjust
    df[f'dots_{dim}_mean_adj'] = np.mean(df[[f'{c}_adj' for c in cols]], axis=1)

    # task
    cols = [f'{dim}_mean_{r}' for r in character_roles_]
    df[[f'{c}_adj' for c in cols]] = df[cols].values - adjust
    df[f'{dim}_mean_mean_adj'] = np.mean(df[[f'{c}_adj' for c in cols]], axis=1)


#-------------------------------------------------------------------------------------
# behavior & perceptions of others (real and task)
# excluding the neutral character rn
#-------------------------------------------------------------------------------------


print('Adding relationship-perception related variables')

# task behavior & dots placements
task_xy    = get_coords(df, 'task', include_neutral=True)
dots_xy    = get_coords(df, 'dots', include_neutral=True)
ratings_xy = (reshape_dataframe(df, [[f'{role}_likability' ,f'{role}_impact' ] for role in character_roles_]) - 50) / 50 # rescale so centered on within -1, +1

# realworld relationships
df.rename(columns={'work_supervision': 'workSupervision', 
                   'work_nonsupervision': 'workNonsupervision',
                   'extra_group1': 'extraGroup1'}, inplace=True)

family        = ['marriage', 'dating', 'children', 'parents', 'relatives', 'inlaws']
nonfamily     = ['friends', 'school', 'neighbors', 'workNonsupervision', 'workSupervision', 'religion', 'volunteer', 'extraGroup1']
relationships = family + nonfamily

# make sure "relationship" is in column name to make easy to find later
for rel in relationships: 
    cols = [c for c in df.columns if (rel in c) & ('demo' not in c)]
    for col in cols:
        if 'relationship' not in col: 
            df.rename(columns={col: f'{col}_relationship'}, inplace=True)

# define the columns for diff. types of relationships
relationship_cols = np.unique([('_').join(c.split('_')[0:2]) for c in df.columns if 'relationship' in c and 'demo' not in c])
nonfamily_cols    = [c for c in relationship_cols if c.split('_')[0] in nonfamily]
family_cols       = [c for c in relationship_cols if c.split('_')[0] in family]
assert len(nonfamily_cols) + len(family_cols) == len(relationship_cols), f'{len(nonfamily_cols)} + {len(family_cols)} != {len(relationship_cols)}'

# reshape data to xy and rescale ratings to [-1, +1]
fam_dots_xy = reshape_dataframe(df, [[f'{rel}_dots_affil_relationship', f'{rel}_dots_power_relationship'] for rel in family_cols])
nonfam_dots_xy = reshape_dataframe(df, [[f'{rel}_dots_affil_relationship', f'{rel}_dots_power_relationship'] for rel in nonfamily_cols])
rw_dots_xy = reshape_dataframe(df, [[f'{rel}_dots_affil_relationship', f'{rel}_dots_power_relationship'] for rel in relationship_cols])

fam_ratings_xy = (reshape_dataframe(df, [[f'{rel}_likability_relationship', f'{rel}_impact_relationship'] for rel in family_cols]) - 50) / 50
nonfam_ratings_xy = (reshape_dataframe(df, [[f'{rel}_likability_relationship', f'{rel}_impact_relationship'] for rel in nonfamily_cols]) - 50) / 50
rw_ratings_xy = (reshape_dataframe(df, [[f'{rel}_likability_relationship', f'{rel}_impact_relationship'] for rel in relationship_cols]) - 50) / 50

# Calculate mean values
mean_columns = {
    'dots_affil_nonfamily_mean': nonfam_dots_xy,
    'dots_power_nonfamily_mean': nonfam_dots_xy,
    'dots_affil_family_mean': fam_dots_xy,
    'dots_power_family_mean': fam_dots_xy,
    'likability_family_mean': fam_ratings_xy,
    'impact_family_mean': fam_ratings_xy,
    'likability_nonfamily_mean': nonfam_ratings_xy,
    'impact_nonfamily_mean': nonfam_ratings_xy,
    # Uncomment the following lines if needed
    # 'likability_relationship_mean': rw_ratings_xy,
    # 'impact_relationship_mean': rw_ratings_xy,
}
for column, data in mean_columns.items():
    df[column] = np.nan
    df[column] = np.nanmean(data, axis=1)

# Add in quadrant variables
role_types = {
    'task': character_roles_,
    'relationship': relationship_cols,
    'nonfamily': nonfamily_cols,
    'family': family_cols
}

for which, coords in {'task_behav': task_xy, 'task_dots': dots_xy, 'task_ratings': ratings_xy, 
                      'relationship_dots': rw_dots_xy, 'relationship_ratings': rw_ratings_xy, 
                      'family_dots': fam_dots_xy, 'family_ratings': fam_ratings_xy,
                      'nonfamily_dots': nonfam_dots_xy, 'nonfamily_ratings': nonfam_ratings_xy}.items():

    roles = role_types[which.split('_')[0]]
    
    # Calculate quadrant for each role
    df[[f'{which}_{role}_quadrant' for role in roles]] = [[get_quadrant(xy[0], xy[1]) for xy in xys] for xys in coords]

    # Calculate quadrant overlap % for each quadrant
    quad_prefs = np.vstack([calc_quadrant_overlap(xys, verbose=False) for xys in coords])
    df[[f'{which}_quadrant{q}_tendency' for q in range(1, 5)]] = quad_prefs # assign to df
    df[f'{which}_quadrants_tendency'] = np.where(np.isnan(quad_prefs).all(axis=1), np.nan, np.argmax(quad_prefs, 1) + 1)

    # Quadrant preferences relative to the other subjects
    quad_relative = calc_quarant_relative(quad_prefs)
    cols = [f'{which}_{col}' for col in quad_relative.columns]
    df.loc[:, cols] = pd.DataFrame(quad_relative.values, columns=cols)

# add in some averages
df['dots_affil_relationship_mean'] = np.nanmean(df[[c for c in df.columns if ('dots' in c) & ('affil' in c) & ('relationship' in c)]], 1)
df['dots_power_relationship_mean'] = np.nanmean(df[[c for c in df.columns if ('dots' in c) & ('power' in c) & ('relationship' in c)]], 1)


#--------------------------------------------------------------------------------------------------
# psych disorders
#--------------------------------------------------------------------------------------------------


disorders = df[['demo_adjust','demo_adhd','demo_autism','demo_bipolar','demo_avpd',
                  'demo_bpd','demo_mdd','demo_ed','demo_gad','demo_ocd','demo_ld',
                  'demo_panic','demo_gambling','demo_ptsd','demo_scz','demo_spd','demo_sad',
                  'demo_tourettes','demo_sud','demo_other_disorder']].fillna(0)
df['disorder'] = (np.sum(disorders, axis=1) > 0) * 1


#--------------------------------------------------------------------------------------------------
# time since lockdown and FDA approval
#--------------------------------------------------------------------------------------------------


# dates = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['days_since_lockdown_start'] = (dates - pd.to_datetime('2020-03-15')).dt.days
# df['days_since_vaccine_fda']    = (dates - pd.to_datetime('2021-8-23')).dt.days # approval
# df['after_vaccine_fda']         = (df['days_since_vaccine_fda'] > 0) * 1


#--------------------------------------------------------------------------------------------------
# add positive control variable
#--------------------------------------------------------------------------------------------------


print('Adding control decision variables')

decision_data = df[[c for c in df.columns if 'decision' in c]].values

# 'prosocial-antisocial'
# larger values -> friendly submissive, smaller values -> unfriendly dominant
df['prosocial_mean_mean'] = np.mean(decision_data, axis=1)
for r, role in enumerate(character_roles_):
    df[f'prosocial_mean_{role}'] = np.mean(decision_data[:, character_labels == r+1], axis=1)

# 'approach-avoid'
# larger values -> friendly dominant, smaller values -> unfriendly submissive
power_mask = decision_trials['dimension'] == 'power'
decision_data[:, power_mask] = decision_data[:, power_mask]  * -1
df['approach_mean_mean'] = np.mean(decision_data, axis=1)
for r, role in enumerate(character_roles_):
    df[f'approach_mean_{role}'] = np.mean(decision_data[:, character_labels == r+1], axis=1)


#--------------------------------------------------------------------------------------------------
# dots variables
#--------------------------------------------------------------------------------------------------


dots_df = pd.DataFrame(columns=['sub_id'])
for r, row in df.iterrows():

    print(f"Adding dots variables: {r+1} / {len(df)}", end='\r')
    dots_df.loc[r, 'sub_id'] = row['sub_id']

    for adj in ['_adj', '']:

        #--------------------------------------------------------------------------------------------------
        # behavior
        #--------------------------------------------------------------------------------------------------

        beh_cols = flatten_lists([[f'affil_mean_{r}{adj}', f'power_mean_{r}{adj}'] for r in character_roles_])
        beh_xy = row[beh_cols].values.astype(float).reshape(-1, 2) # scaled to [-1, 1]
        beh_pov_dists = np.array([norm(xy-[1,0]) for xy in beh_xy]).astype(float).reshape(-1, 1) # from pov

        #--------------------------------------------------------------------------------------------------
        # dots subjective placements
        #--------------------------------------------------------------------------------------------------

        dots_cols = flatten_lists([[f'{r}_dots_affil{adj}', f'{r}_dots_power{adj}'] for r in character_roles_])
        dots_xy   = row[dots_cols].values.astype(float).reshape(-1, 2)
        if np.isnan(dots_xy).any(): continue

        dots_df.loc[r, [f'dots_affil_mean{adj}', f'dots_power_mean{adj}']] = np.mean(dots_xy, axis=0)
        dots_df.loc[r, [f'dots_surface_area{adj}', f'dots_area{adj}']] = ComputeBehavior2.calc_shape_size(dots_xy)
        dots_df.loc[r, f'dots_avg_pw_dist{adj}'] = np.mean(get_rdv(dots_xy), axis=0)

        # area of convex hull
        dots_df.loc[r, 'dots_area'] = calc_area(dots_xy)

        # distance from self
        dots_pov_dists = np.array([norm(xy-[1, 0]) for xy in dots_xy]).astype(float).reshape(-1, 1) # euclidean distances from pov to xy
        dots_df.loc[r, [f'dots_pov_dist_{r}{adj}' for r in character_roles_]] = dots_pov_dists.flatten()
        dots_df.loc[r, f'dots_pov_dist_mean{adj}'] = np.mean(dots_pov_dists, axis=0)

        #--------------------------------------------------------------------------------------------------
        # behavior vs. dots
        #--------------------------------------------------------------------------------------------------

        # distances between behavior and dots
        beh_dots_dists = norm(beh_xy - dots_xy, axis=1)
        dots_df.loc[r, [f'beh_dots_dist_{r}{adj}' for r in character_roles_]] = beh_dots_dists.flatten()
        dots_df.loc[r, f'beh_dots_dist_mean{adj}'] = np.mean(beh_dots_dists, axis=0)

        beh_dots_pov_dists = norm(beh_pov_dists - dots_pov_dists, axis=1) # from POV
        dots_df.loc[r, [f'beh_dots_pov_dist_diff_{r}{adj}' for r in character_roles_]] = beh_dots_pov_dists.flatten()
        dots_df.loc[r, f'beh_dots_pov_dist_diff_mean{adj}'] = np.mean(beh_dots_pov_dists, axis=0)

        # single dimension distances
        xy_diff = beh_xy - dots_xy
        dots_df.loc[r, [f'beh_dots_affil_diff_{r}{adj}' for r in character_roles_]] = xy_diff[:,0]
        dots_df.loc[r, f'beh_dots_affil_diff_mean{adj}'] = np.mean(xy_diff[:,0])

        dots_df.loc[r, [f'beh_dots_power_diff_{r}{adj}' for r in character_roles_]] = xy_diff[:,1]
        dots_df.loc[r, f'beh_dots_power_diff_mean{adj}'] = np.mean(xy_diff[:,1])


        # are permuted locations further away from the behavior than the real dots locations?
        dots_shuff = [np.mean(norm(beh_xy - np.random.permutation(dots_xy), axis=1)) for _ in range(1000)]
        dots_df.loc[r, f'beh_dots_dist_shuff_mean{adj}'] = np.mean(dots_shuff)
        dots_df.loc[r, f'beh_dots_dist_shuff_std{adj}']  = np.std(dots_shuff)

        # RSAs for behavior and dots
        dots_df.loc[r, f'beh_dots_pw_dist_tau{adj}']  = kendalltau(get_rdv(dots_xy), get_rdv(beh_xy))[0]
        dots_df.loc[r, f'beh_dots_pov_dist_tau{adj}'] = kendalltau(get_rdv(dots_pov_dists), get_rdv(beh_pov_dists))[0]

        # procrustes transformation: rotationm reflection, scaling & translation
        # - disparity = sum of the squares of the point-wise euclidean distances
        dots_df.loc[r, 'beh_dots_procrustes_disp'] = scipy.spatial.procrustes(dots_xy, beh_xy)[2] # order doesnt matter

        # orthogonal procrustes: rotation & reflection (no scaling or translation)
        dots_df.loc[r, 'beh_dots_orth_procrustes_scale'] = scipy.linalg.orthogonal_procrustes(beh_xy, dots_xy)[1] # order **does** matter: (to_map, map_to)

        # least squares: task @ transform = dots
        try: 
            _, res, _ , _ = np.linalg.lstsq(beh_xy, dots_xy, rcond=None)
            dots_df.loc[r, ['beh_dots_lstsqres_affil', 'beh_dots_lstsqres_power']] = res[0], res[1]
        except:
            dots_df.loc[r, ['beh_dots_lstsqres_affil', 'beh_dots_lstsqres_power']] = np.nan, np.nan
        

        #--------------------------------------------------------------------------------------------------
        # memory representational similarity analysis
        #--------------------------------------------------------------------------------------------------


        dists_diff_rdv = get_rdv(beh_dots_dists.reshape(-1,1)) # distance between dots and task as an rdv
        memory_rdv = get_rdv(row[[f'memory_{r}' for r in character_roles_]].values.reshape(-1,1))
        dots_df.loc[r, f'beh_dots_dist_memory_tau{adj}'] = kendalltau(dists_diff_rdv, memory_rdv)[0]
    
# convert data types and update df
dots_df.iloc[:,1:] = dots_df.iloc[:,1:].astype(float)
df.loc[:, dots_df.columns] = dots_df


#--------------------------------------------------------------------------------------------------
# character ratings
#--------------------------------------------------------------------------------------------------


print('Adding character ratings')

# get the difference from self rating
ratings = ['friendliness', 'competence', 'dominance', 'popularity', 'likability', 'impact']
for rating, role in itertools.product(ratings, character_roles_):
    df[f'{role}_{rating}_diff'] = df[f'{role}_{rating}'] - df[f'self_{rating}']

# calculate means
for rating in ratings: 
    df[f'{rating}_mean']      = np.nanmean(df[[f'{role}_{rating}' for role in character_roles_]],1)
    df[f'{rating}_diff_mean'] = np.nanmean(df[[f'{role}_{rating}_diff' for role in character_roles_]],1)


#------------------------------------------------------------------------
# SNT prospective choice
#------------------------------------------------------------------------


print('Adding prospective choice variables')

# each character 
for role in character_roles_:
    choice = df[[c for c in df.columns if ('forced_choice' in c) & (f'_{role}_' in c) & ('reaction_time' not in c)]]
    choice = choice.loc[:, [x==role for x in [c.split('_')[-1] for c in choice.columns]]]
    df[f'prospective_{role}_mean'] = np.mean(choice, axis=1)

# trial by trial preference
choices = df[[c for c in df.columns if ('forced_choice' in c) & ('reaction_time' not in c)]]
choice_comps = np.unique([('_').join(c.split('_')[2:5]) for c in choices.columns])

for comp in choice_comps:
    chars = comp.split('_v_')

    # separate subjs who chose the 1st or 2nd character
    mask01 = (df[f'forced_choice_{comp}_{chars[0]}'] > df[f'forced_choice_{comp}_{chars[1]}'])
    mask02 = (df[f'forced_choice_{comp}_{chars[0]}'] < df[f'forced_choice_{comp}_{chars[1]}'])

    pref, affil, power, dist, cos = [], [], [], [], [] 
    for m, mask in enumerate([mask01, mask02]):
        pc, npc = (0, 1) if m == 0 else (1, 0)
        pref.append((df[f'forced_choice_{comp}_{chars[pc]}'] * (mask * 1)))
        affil.append(df[f'affil_mean_{chars[pc]}'][mask] - df[f'affil_mean_{chars[npc]}'][mask])
        power.append(df[f'power_mean_{chars[pc]}'][mask] - df[f'power_mean_{chars[npc]}'][mask])
        dist.append(df[f'pov_2d_dist_{chars[pc]}'][mask] - df[f'pov_2d_dist_{chars[npc]}'][mask])
        cos.append(np.cos(df[f'pov_2d_angle_{chars[pc]}'][mask]) - np.cos(df[f'pov_2d_angle_{chars[npc]}'][mask]))

    # preferred character, preference amount, affil diff, power diff 
    pref_char = []
    for t in (mask01.values * 1) + (mask02.values * 2):
        if t == 1:   pref_char.append(chars[0])
        elif t == 2: pref_char.append(chars[1])
        else:        pref_char.append(np.nan)

    df[f'prospective_char_pref_{comp}']   = pref_char
    df[f'prospective_choice_pref_{comp}'] = pref[0].values + pref[1].values # should all be > 0
    df[f'prospective_choice_rt_{comp}']   = df[f'forced_choice_{comp}_reaction_time']
    df[f'prospective_affil_pref_{comp}']  = pd.concat([affil[0], affil[1]]) 
    df[f'prospective_power_pref_{comp}']  = pd.concat([power[0], power[1]])
    df[f'prospective_dist_pref_{comp}']   = pd.concat([dist[0], dist[1]])
    df[f'prospective_cos_pref_{comp}']    = pd.concat([cos[0], cos[1]])

    # check if being diff skincolor/gender matters
    df[f'prospective_skincolor_{comp}']   = (df[f'{chars[0]}_gender']    == df[f'{chars[1]}_gender']) * 1
    df[f'prospective_gender_{comp}']      = (df[f'{chars[0]}_skincolor'] == df[f'{chars[1]}_skincolor']) * 1

# average across characters
df['prospective_choice_rt_avg']   = np.nanmean(df[[c for c in df.columns if 'prospective_choice_rt_' in c]], axis=1)
df['prospective_choice_pref_avg'] = np.nanmean(df[[c for c in df.columns if 'prospective_choice_pref_' in c]],axis=1)



#--------------------------------------------------------------------------------------------------
# standardize some variables, for convenience later
#--------------------------------------------------------------------------------------------------

# cosine & sine
angle_preds = ['neu_2d_angle_mean', 'neu_2d_angle_mean_mean', 'pov_2d_angle_mean', 
              'pov_2d_angle_mean_mean', 'neu_3d_angle_mean', 'pov_3d_angle_mean']
for pred in angle_preds:
    df[f'{pred}_cos'] = np.cos(df[f'{pred}'])
    df[f'{pred}_sin'] = np.sin(df[f'{pred}'])
angle_preds = [f'{p}_cos' for p in angle_preds] + [f'{p}_sin' for p in angle_preds]


# save
print('\n--------------------------------------------------')
print('SAVING DATA')
print('--------------------------------------------------')

df.to_excel(f'{base_dir}/Data/All-data_summary_n{len(df)}.xlsx', index=False) # overwrite w/ new variables