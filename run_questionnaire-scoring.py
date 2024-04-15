#---------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import glob
from datetime import date
import math

def subset_df(df, prefixes):
    subset_dfs = []
    df = df[[c for c in df if 'score' not in c and 'att' not in c]].copy()
    for prefix in prefixes:
        ixs = [c for c, col in enumerate(df.columns) if col.startswith(prefix)] 
        subset_dfs.append(df.iloc[:,ixs])
    subset_df = pd.concat(subset_dfs, axis=1)
    return subset_df

def reverse_score(scores_df, high_score, rev_items):
    '''Reverse scores: e.g., 1,2,3,4 -> 4,3,2,1
         high_score = highest score available
         rev_items = 0-ixed array of items to reverse 
         returns: full array of all values, easy to split up into subscales etc
     '''
    scores = scores_df.values
    if high_score > 1:
        scores[:,rev_items] = (high_score+1) - scores[:,rev_items]
    elif high_score == 1:    
        scores[:,rev_items] = (scores[:,rev_items] == 0)*1
    return pd.DataFrame(scores, columns=scores_df.columns.values)

def score_bapq(df):
    bapq_rev = np.array([1,3,7,9,12,15,16,19,21,23,25,28,30,34,36])-1
    bapq_df_ = subset_df(df, ['bapq'])
    bapq_df = reverse_score(bapq_df_, 6, bapq_rev)
    bapq_df['bapq_score'] = np.sum(bapq_df, 1).values
    bapq_df['bapq_aloof_score'] = np.sum(bapq_df.iloc[:,np.array([1,5,9,12,16,18,23,25,27,28,31,36])-1],1).values
    bapq_df['bapq_prag_lang_score'] = np.sum(bapq_df.iloc[:,np.array([2,4,7,10,11,14,17,20,21,29,32,34])-1],1).values
    bapq_df['bapq_rigid_score'] = np.sum(bapq_df.iloc[:,np.array([3,6,8,13,15,19,22,24,26,30,33,35])-1],1).values
    
    return bapq_df

def score_sni(df):

    sni_items = df[[c for c in raw_df.columns if 'sni' in c]]
    sni_items = sni_items.fillna(0)
    sni_scores = np.zeros((len(df),3))
    
    # number of high-contact roles
    sni_scores[:,0] += (sni_items.loc[:,'sni_1'] == 1) * 1
    for item in ['2a','3a','4a','5a','6a','7a','8a','10','11a']:
        sni_scores[:,0] += (sni_items.loc[:,'sni_'+item] > 0) * 1
    for n in np.arange(1,8):
        sni_scores[:,0] += (sni_items.loc[:,'sni_12a' + str(n) + 'number'] > 1) * 1
    sni_scores[:,0] += (sni_items.loc[:, 'sni_9a'] > 0) & (sni_items.loc[:, 'sni_9b'] > 0) * 1

    # number of people in social network
    sni_scores[:,1] += (sni_items.loc[:,'sni_1'] == 1) * 1
    sni_scores[:,1] += sni_items.loc[:,'sni_2a']
    sni_items.loc[:,'sni_3a'][sni_items.loc[:,'sni_3a'] == 3] = 2
    sni_items.loc[:,'sni_3a'][((sni_items.loc[:,'sni_3a'] == 1) | (sni_items.loc[:,'sni_3a'] == 2))] = 1 # 1 or 2 -> 1
    sni_items.loc[:,'sni_4a'][sni_items.loc[:,'sni_4a'] == 3] = 2
    sni_items.loc[:,'sni_4a'][((sni_items.loc[:,'sni_4a'] == 1) | (sni_items.loc[:,'sni_4a'] == 2))] = 1 # 1 or 2 -> 1
    for item in ['3a','4a','5a','9a','9b','10','11a']:
        sni_scores[:,1] += sni_items.loc[:,'sni_'+item]
    for n in np.arange(1,8):
        sni_scores[:,1] += sni_items.loc[:,'sni_12a' + str(n) + 'number']

    # number of embedded networks
    sni_scores[:,2] += (np.sum(sni_items.loc[:,['sni_1','sni_2a','sni_3a','sni_4a','sni_5a']], axis=1) > 4) * 1
    for item in ['6a','7a','8a','10','11a']:
        sni_scores[:,2] += (sni_items.loc[:,'sni_'+ item] > 4)
    sni_scores[:,2] += (sni_items.loc[:,'sni_9a'] + sni_items.loc[:,'sni_9b'] > 4) * 1
    for n in np.arange(1,8):
        sni_scores[:,2] += (sni_items.loc[:,'sni_12a' + str(n) + 'number'] > 4) * 1

    sni_scores = pd.DataFrame(sni_scores, columns=['sni_hc_score', 'sni_size_score', 'sni_emb_score'])
    sni_df = pd.concat([sni_items, sni_scores], axis=1)
    return sni_df

def score_sss(df):
    
    sss_items = ['sss_' + str(i) for i in range(1,44)]
    sss_df = reverse_score(df[sss_items].copy(), 1, sss_rev)
    
    # summarize
    sss_df['sss_att'] = df['sss_att']
    sss_df['sss_score'] = np.sum(sss_df, axis=1)
    sss_df['sss_unus_exp_score'] = np.sum(sss_df.iloc[:,np.arange(1,13)-1],axis=1)
    sss_df['sss_cog_dis_score'] = np.sum(sss_df.iloc[:,np.arange(13,24)-1],axis=1)
    sss_df['sss_intro_anhe_score'] = np.sum(sss_df.iloc[:,np.arange(24,34)-1],axis=1)
    sss_df['sss_impuls_noncon_score'] = np.sum(sss_df.iloc[:,np.arange(34,44)-1],axis=1)
    
    return sss_df

def score_eat(df):

    eat_items = ['eat_' + str(i) for i in range(1,27)]
    dieting_items = np.array([1,6,7,10,11,12,14,16,17,22,23,24,26])-1
    preoccup_items = np.array([3,4,9,18,21,25])-1
    control_items = np.array([2,5,8,13,15,19,20])-1

    eat_df = df[eat_items].copy()
    
    # rescore stuff
    eat_df[eat_df.iloc[:,0:25] > 3] = 0
    eat_df[eat_df.iloc[:,0:25] == 1] = 3
    eat_df[eat_df.iloc[:,0:25] == 3] = 1
    eat_df[eat_df.iloc[:,25] < 3] = 0
    eat_df[eat_df.iloc[:,25] == 4] = 1
    eat_df[eat_df.iloc[:,25] == 5] = 2
    eat_df[eat_df.iloc[:,25] == 6] = 3
    
    # summarize
    eat_df['eat_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_dieting_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_bulmia_food_preoc_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_oral_control_score'] = np.sum(eat_df.iloc[:,control_items],axis=1)
    
    return eat_df

def score_pid(df):
    
    neg_aff = np.array([8,9,10,11,15])-1
    detach = np.array([4,13,14,16,18])-1
    antagonism = np.array([17,19,20,22,25])-1
    disinhibtion = np.array([1,2,3,5,6])-1
    psychotism = np.array([7,12,21,23,24])-1

    pid_df = df[['pid5_' + str(i) for i in range(1,26)]]
    pid_df['pid5_neg_aff_score'] = np.sum(pid_df.iloc[:,neg_aff], axis=1)
    pid_df['pid5_detachment_score'] = np.sum(pid_df.iloc[:,detach], axis=1)
    pid_df['pid5_antagonism_score'] = np.sum(pid_df.iloc[:,antagonism], axis=1)
    pid_df['pid5_disinhibiton_score'] = np.sum(pid_df.iloc[:,disinhibtion], axis=1)
    pid_df['pid5_psychotism_score'] = np.sum(pid_df.iloc[:,psychotism], axis=1)
    
    return pid_df

# reverse items: -1 to account for python 0 indexing
stai_s_rev  = np.array([1,2,5,8,10,11,15,16,19,20])-1
stai_t_rev = np.array([1,3,6,7,10,13,14,16,19])-1
#from sarah's msp: 1_r,3_r,6_r,7_r,10_r,13_r,14_r,16_r,19_r,
sds_rev = np.array([2,5,6,11,12,14,16,17,18,20])-1
aes_rev = np.array([1,2,3,4,5,7,8,9,12,13,14,15,16,17,18])-1
sss_rev = np.array([26,27,28,30,31,34,37,39])-1
eat_rev  = np.array([26])-1
bapq_rev  = np.array([1,3,7,9,12,15,16,19,21,23,25,28,30,34,36])-1

data_dir = '/Users/matty_gee/Desktop/SNT/SNT-behavior/Online/Prolific/Data/Original_2021/Questionnaires'

#---------------------------------------------------------------------------------------------------------------------------------------------------------

questionnaire_csvs = glob.glob(data_dir + "/Raw_questionnaire_data/*.csv")

# subs to ignore
sess_id_ = '6068c71605acb9a48922983a'
pid_excl = [ '5de5538f8fde1c4dbc951498','5d6141608a11df001a477fbe', '602f08507cdd70bc977a302e',
             '5e7913d548988f0c5bd98567','5fb28e64134ec0894396633a',
             '5e532bad158f1d33b3473072','5bf17270871dd1000197226e','5e69be2cd02ab2027bff5108']

csv_dfs = []
for csv in questionnaire_csvs: 
    
    raw_df = pd.read_csv(csv)
    raw_df = raw_df[[c for c in raw_df if 'complete' not in c and 'timestamp' not in c]] # filter these cols out...

    redcap_ver = int(csv.split('/')[-1].split('_')[0][-1])

    # check if sub is to be excluded:
    sub_ids = raw_df['prolific_pid'].values
    inclusion_list = []
    for sub in sub_ids:
        if type(sub) is not str:
            if math.isnan(sub):
                inclusion_list.append(False)
        elif sub in pid_excl:
            inclusion_list.append(False)
        else:
            inclusion_list.append(True)
    raw_df = raw_df.iloc[np.array(inclusion_list),:]
    raw_df.reset_index(drop=True, inplace=True)
    sub_ids = sub_ids[inclusion_list]

    ################################################
    # DEMOS
    ################################################

    demo_df = subset_df(raw_df, ['demo'])
    demo_df.rename(columns = {'demo_race___1': 'demo_race_amer_indian_or_alaska_native',
                             'demo_race___2': 'demo_race_asian',
                             'demo_race___3': 'demo_race_black/aa',
                             'demo_race___4': 'demo_race_latino/hispanic',
                             'demo_race___5': 'demo_race_multiracial',
                             'demo_race___6': 'demo_race_nat_hawaiian_or_pac_isl',
                             'demo_race___7': 'demo_race_white',
                             'demo_race___8': 'demo_race_other',
                             'demo_race_other': 'demo_race_other_text',
                             'demo_sex': 'demo_sex_1F',
                             'demo_gender': 'demo_gender_1W',
                             'demo_handedness': 'demo_handedness_1R'}, 
                    inplace = True)

    ################################################
    # COV19 RESPS
    ################################################

    cov19_df = subset_df(raw_df, ['cov19']) 

    ################################################
    # OCI 
    ################################################

    ## SCORING DETAILS: just sum
    
    oci_df = subset_df(raw_df, ['oci']) # think can just subset the df normally...?
    oci_df['oci_score'] = np.sum(oci_df, axis=1)
    oci_df['oci_att']   = raw_df['oci_att']

    ################################################
    # STAI STATE 
    ################################################

    ## SCORING DETAILS: some are reversed (4,3,2,1): 1,2,5,8,10,11,15,16,19,20
    
    stai_s_df_ = subset_df(raw_df, ['stai_s'])
    stai_s_df  = reverse_score(stai_s_df_, 4, stai_s_rev)
    stai_s_df['stai_s_score'] = np.sum(stai_s_df, axis=1)
    stai_s_df['stai_s_att']   = raw_df['stai_s_att']

    ################################################
    # ZUNG SDS 
    ################################################
    
    ## SCORING DETAILS: some are reversed (4,3,2,1): 2,5,6,11,12,14,16,17,18,20
    
    sds_df_   = subset_df(raw_df, ['sds'])
    sds_df    = reverse_score(sds_df_, 4, sds_rev)
    sds_df['sds_score'] = np.sum(sds_df, axis=1)
    sds_df['sds_att']   = raw_df['sds_att']

    ################################################
    # AQ 
    ################################################

    ## SCORING DETAILS:
    # 1= def agree, 2= slightly agree, 3= slightly disagree, 4=def disagree
    # "Definitely agree" (1) or "Slightly agree" (2) = 1: questions: 2,4, 5, 6, 7, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46
    # "Definitely disagree" (3) or "Slightly disagree" (4) = 1: questions: 1, 3, 8, 10, 11, 14, 15, 17, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 44, 47, 48, 49, 50
    
    aq_df_  = subset_df(raw_df, ['aq'])

    aq_items = np.append(np.array([raw_df['aq_' + str(q)].values for q in np.array((2,4,5,6,7,9,12,13,16,18,19,20,21,22,23,26,33,35,39,41,42,43,45,46))]) <= 2,
               np.array([raw_df['aq_' + str(q)].values for q in np.array((1,3,8,10,11,14,15,17,24,25,27,28,29,30,31,32,34,36,37,38,40,44,47,48,49,50))]) > 2, axis=0).T*1

    # subscales
    aq_social    = np.array([1,11,13,15,22,36,44,45,47,48])
    aq_switching = np.array([2,4,10,16,25,32,34,37,43,46])
    aq_details   = np.array([5,6,9,12,19,23,28,29,30,49])
    aq_commn     = np.array([7,17,18,26,27,31,33,35,38,39])
    aq_imagin    = np.array([3,8,14,20,21,24,40,41,42,50])

    # rename
    aq_cols = []
    for q in range(1,51):
        if q in aq_social:
            newname = 'aq_social_'+str(q)
        elif q in aq_switching:
            newname = 'aq_switching_'+str(q)
        elif q in aq_details:
            newname = 'aq_details_'+str(q)
        elif q in aq_commn:
            newname = 'aq_commn_'+str(q)
        elif q in aq_imagin:
            newname = 'aq_imagin_'+str(q)
        aq_cols.append(newname)

    aq_df = pd.DataFrame(aq_items, columns=aq_cols)
    aq_df['aq_social_score']    = np.sum(aq_items[:,aq_social-1],axis=1) # account for 0-idxing 
    aq_df['aq_switching_score'] = np.sum(aq_items[:,aq_switching-1],axis=1)
    aq_df['aq_details_score']   = np.sum(aq_items[:,aq_details-1],axis=1)
    aq_df['aq_commn_score']     = np.sum(aq_items[:,aq_commn-1],axis=1)
    aq_df['aq_imagn_score']     = np.sum(aq_items[:,aq_imagin-1],axis=1)
    aq_df['aq_score']           = np.sum(aq_items, axis=1)
    aq_df['aq_att']             = raw_df['aq_att']

    ################################################
    # AUDIT
    ################################################

    ## SCORING DETAILS: just sum
    
    audit_df = subset_df(raw_df, ['audit'])
    audit_df['audit_score'] = np.sum(np.array([raw_df['audit_' + str(q)].values for q in np.arange(1,11)]), axis=0)
    audit_df['audit_att']   = raw_df['audit_att']
    
    ################################################
    # AES
    ################################################

    ## SCORING DETAILS 
    # create new df w/ different scoring from what redcap outputted
    # some are reversed (4,3,2,1): 1,2,3,4,5,7,8,9,12,13,14,15,16,17,18
    
    aes_raw = raw_df[['aes_' + str(s) for s in np.arange(1,19)]].values
    aes_raw[aes_raw == 3] = 4
    aes_raw[aes_raw == 2] = 3
    aes_raw[aes_raw == 0] = 1
    aes_df_ = pd.DataFrame(aes_raw, columns=raw_df[['aes_' + str(s) for s in np.arange(1,19)]].columns.values)
    
    aes_df  = reverse_score(aes_df_, 4, aes_rev)
    aes_df['aes_score'] = np.sum(aes_df, axis=1)
    aes_df['aes_cog_score'] = np.sum(aes_df.iloc[:,np.array([1,3,4,5,8,11,12,15])-1], axis=1)
    aes_df['aes_beh_score'] = np.sum(aes_df.iloc[:,np.array([2,6,9,10,12])-1], axis=1)
    aes_df['aes_emt_score'] = np.sum(aes_df.iloc[:,np.array([7,14])-1], axis=1)
    aes_df['aes_oth_score'] = np.sum(aes_df.iloc[:,np.array([15,17,18])-1], axis=1)
    aes_df['aes_att'] = raw_df['aes_att'] 
    
    csv_df = pd.concat((demo_df, cov19_df, oci_df, audit_df, stai_s_df, aes_df, sds_df, aq_df), axis=1)

    ################################################
    # check if these exist in the sheet
    ################################################    
    # could be in this csv but could be in one of the others
    # EAT, LSAS, APDIS, BPD, SSS, PDI, SNI, PID, BAPQ
    
#     other_df = pd.DataFrame()
    if 'sss_1' in raw_df.columns:
        sss_df = score_sss(raw_df) # includes items
        csv_df = pd.concat((csv_df, sss_df), axis=1)
        
    if 'eat_1' in raw_df.columns:  
        eat_df = score_eat(raw_df) # includes items
        csv_df = pd.concat((csv_df, eat_df), axis=1)
        
    if 'lsas_av_1' in raw_df.columns:
        lsas_df = raw_df[['lsas_av_score', 'lsas_av_att'] + ['lsas_av_' + str(i) for i in np.arange(1,25)]]
        csv_df = pd.concat((csv_df, lsas_df), axis=1)
        
    if 'apdis_score' in raw_df.columns:
        apdis_df = raw_df[['apdis_score', 'apdis_att'] + ['apdis_' + str(i) for i in np.arange(1,9)]]
        csv_df = pd.concat((csv_df, apdis_df), axis=1)
        
    if 'zbpd_score' in raw_df.columns:
        zbpd_df = raw_df[['zbpd_score', 'zbpd_att'] + ['zbpd_' + str(i) for i in np.arange(1,11)]]
        csv_df = pd.concat((csv_df, zbpd_df), axis=1)
        
    if 'pid5_1' in raw_df.columns:   
        pid_df = score_pid(raw_df) # includes items
        csv_df = pd.concat((csv_df, pid_df), axis=1)
        
    if 'sni_1' in raw_df.columns:   
        sni_df = score_sni(raw_df) # includes items
        csv_df = pd.concat((csv_df, sni_df), axis=1)
        
    if 'pdi_1' in raw_df.columns:
        pdi_df = pd.DataFrame()
        pdi_df['pdi_raw_score'] = raw_df['pdi_raw_score']
        pdi_df['pdi_distress_score'] = raw_df['pdi_distress_score']
        pdi_df['pdi_think_score'] = raw_df['pdi_think_score']
        pdi_df['pdi_believe_score'] = raw_df['pdi_believe_score']
        pdi_df['pdi_total_score'] = raw_df['pdi_total_score']
        pdi_df['pdi_att'] = raw_df['pdi_att']
        csv_df = pd.concat((csv_df, pdi_df), axis=1)
        
    if 'bapq_1' in raw_df.columns:
        bapq_df = score_bapq(raw_df) # includes items
        csv_df = pd.concat((csv_df, bapq_df), axis=1)
        
    ################################################
    # put all together 
    ################################################
    
    csv_df.insert(0, 'sub_id', sub_ids)
    csv_df.insert(1, 'redcap_ver', redcap_ver)
    csv_dfs.append(csv_df)
    
main_df = pd.concat(csv_dfs, axis=0)
my_subs = main_df['sub_id'].values


tp1_df = pd.read_excel(data_dir + '/Gu_Lab/Prolific_Covid_week1.xlsx')
scz_df = pd.read_excel(data_dir + '/Gu_Lab/Prolific_SCZ_Clean.xlsx')
msp_df = pd.read_excel(data_dir + '/Gu_Lab/Prolific_Misophonia_Clean.xlsx')

##########################################
# COVID WK 1: check - these should already be reverse coded...right?
# - collected older versions of some: sds, oci, stai_s <- exclude these bc we have better
##########################################

# STAI TRAIT 
stai_t_df = reverse_score(tp1_df[['stai_t_' + str(i) for i in range(1,21)]], 4, stai_t_rev)
stai_t_df['stai_t_score'] = np.sum(stai_t_df, axis=1)
stai_t_df.insert(0, 'sub_id', tp1_df['prolific'].values)

# get things that werent collected in my timepoint:
rem_vars = ['ucls', 'dtm', 'dtn', 'dtp', 'ace', 'talc']
rem_tp1_df = subset_df(tp1_df, rem_vars) 
rem_tp1_df[[v + '_score' for v in rem_vars]] = tp1_df[[v + '_score' for v in rem_vars]]
rem_tp1_df[['sd_score','demo_polit_party', 'demo_zip_code']] = tp1_df[['sd_total','polit_party', 'demo_zip_code']]

tp1_df = pd.concat((stai_t_df, rem_tp1_df),axis=1)


##########################################
# SCZ COLLECTION
##########################################

# PDI & PQ16 do not have reverse items
# O-life/SSS has some reverse items - I have to calculate & rescore

# SSS 
sss_df = score_sss(scz_df)
sss_df.insert(0, 'sub_id', scz_df['prolific_pid'].values)

# PQ16 
pq16_df = scz_df[['cpu2_pq16_sc_' + str(i) for i in range(1,33)]]
pq16_df['cpu2_pq16_raw_score'] = scz_df['cpu2_pq16_raw_score']
pq16_df['cpu2_pq16_total_score'] = scz_df['cpu2_pq16_total_score']
pq16_df['cpu2_pq16_distress_score'] = scz_df['cpu2_pq16_distress_score']
pq16_df.columns = [col.replace('cpu2_','') for col in pq16_df.columns]
pq16_df.columns = [col.replace('cpu2_','') for col in pq16_df.columns]

# PDI
pdi_df = scz_df[['pdi_' + str(i) for i in range(1,22)]]
pdi_df['pdi_raw_score'] = scz_df['pdi_raw_score']
pdi_df['pdi_distress_score'] = scz_df['pdi_distress_score']
pdi_df['pdi_think_score'] = scz_df['pdi_think_score']
pdi_df['pdi_believe_score'] = scz_df['pdi_believe_score']
pdi_df['pdi_total_score'] = scz_df['pdi_total_score']
pdi_df['pdi_att'] = scz_df['pdi_att']

scz_df = pd.concat((sss_df, pdi_df, pq16_df), axis=1)

##########################################
# misophonia: these should already be reverse coded
##########################################

msp_df.rename(columns={'prolific_id':'sub_id','dieting':'eat_dieting_score',
                       'bulmia_food_preoc':'eat_bulmia_food_preoc_score',
                       'oral_control':'eat_oral_control_score'}, inplace=True)

##########################################
# output
##########################################

other_df = tp1_df.merge(scz_df, how='outer', on='sub_id').merge(msp_df, how='outer', on='sub_id')
other_df = other_df.drop_duplicates(subset=['sub_id'])
other_df.reset_index(drop=True, inplace=True)
main_df.reset_index(drop=True, inplace=True)

other_df = other_df[other_df['sub_id'].isin(list(main_df['sub_id']))]
other_df = other_df.sort_values(by='sub_id', ascending=False)
other_df.reset_index(drop=True, inplace=True)
main_df = main_df.sort_values(by='sub_id', ascending=False)
main_df.reset_index(drop=True, inplace=True)

overlap = set(main_df.columns[1:]) & set(other_df.columns[1:])
merged_df = pd.concat([main_df, other_df],axis=1)
for col in overlap:
    ix = np.where(merged_df.columns.get_loc(col) == True)[0][0]
    summed = np.sum(merged_df[col],axis=1)
    del merged_df[col]
    merged_df.insert(int(ix), col, summed)

merged_df.to_excel('/Users/matty_gee/Desktop/SNT/SNT-behavior/Online/Prolific/Data/Original_2021/Summary/Questionnaire_summary_n' + str(len(merged_df)) + '.xlsx', index=False)