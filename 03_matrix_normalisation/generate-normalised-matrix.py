#!/bin/python


# ----------------- #
# LOAD DEPENDENCIES
# ----------------- #

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


grandparent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(grandparent_dir)
from functions import normalising


# ----------------- #
# IMPORT R-NORMALISED MATRIX 
# ----------------- #

matrix = pd.read_csv('/data/home/bty449/ExplainableAI/MatrixCSVs/NBA-Matrix_Quantile.csv', header = 0)

matrix = matrix.set_index('phosphosite_ID')

dataset_list = normalising.create_dataframe_per_dataset(matrix)
print(f'Dataframe has been created per dataset')



# ----------------- #
# LOAD DATASET NAMES
# ----------------- #

# define names for each dataset
MC2020_names = ('MC2020_6h_rep1', 'MC2020_6h_rep2', 'MC2020_6h_rep3',
                        'MC2020_12h_rep1', 'MC2020_12h_rep2', 'MC2020_12h_rep3', 
                        'MC2020_20h_rep1', 'MC2020_20h_rep2', 'MC2020_20h_rep3', 
                        'MC2020_48h_rep1', 'MC2020_48h_rep2', 'MC2020_48h_rep3')

TK2018_names = ('TK2018_L/Hratio_t0', 'TK2018_L/Hratio_t5', 'TK2018_L/Hratio_t10',
                        'TK2018_L/Hratio_t15', 'TK2018_L/Hratio_t20', 'TK2018_L/Hratio_t25', 
                        'TK2018_L/Hratio_t30', 'TK2018_L/Hratio_t35', 'TK2018_L/Hratio_t40', 
                        'TK2018_L/Hratio_t45')

RP2022_names = ('RP2022_126_30min', 'RP2022_127n_30min', 'RP2022_127c_30min',
                        'RP2022_128n_40min', 'RP2022_128c_40min', 'RP2022_129n_40min', 
                        'RP2022_129c_noco', 'RP2022_130n_noco', 'RP2022_130c_noco')

RP2023_names = ('RP2023_centrifugation400_rep1', 'RP2023_centrifugation400_rep2', 
                'RP2023_centrifugation400_rep3', 'RP2023_centrifugation4000_rep1',
                'RP2023_centrifugation4000_rep2', 'RP2023_centrifugation4000_rep3', 
                'RP2023_filtration_rep1', 'RP2023_filtration_rep2', 'RP2023_filtration_rep3')

LY2021_names = ('LY2021_L/HratioNormalised_G1_rep1', 'LY2021_L/HratioNormalised_G1_rep2', 
                        'LY2021_L/HratioNormalised_G1_rep3', 'LY2021_L/HratioNormalised_G1_rep4',
						'LY2021_L/HratioNormalised_G2M', 'LY2021_L/HratioNormalised_MMS_rep1', 
                        'LY2021_L/HratioNormalised_MMS_rep2', 'LY2021_L/HratioNormalised_MMS_rep3',
						'LY2021_L/HratioNormalised_MMS_rep4', 'LY2021_L/HratioNormalised_S_rep1',
						'LY2021_L/HratioNormalised_S_rep2')

VB2014_names = ('VB2014_Pherom0NaCl0_average', 'VB2014_Pherom0NaCl1_average', 
                        'VB2014_Pherom0NaCl5_average', 'VB2014_Pherom0NaCl10_average',
						'VB2014_Pherom0NaCl20_average', 'VB2014_Pherom0NaCl45_average', 
                        'VB2014_Pherom1NaCl0_average', 'VB2014_Pherom1NaCl1_average',
						'VB2014_Pherom1NaCl5_average', 'VB2014_Pherom1NaCl10_average',
						'VB2014_Pherom1NaCl20_average', 'VB2014_Pherom1NaCl45_average', 
                        'VB2014_Pherom5NaCl0_average', 'VB2014_Pherom5NaCl1_average', 
                        'VB2014_Pherom5NaCl15_average')

KK2015score_names = ('KK2015_score_T00', 'KK2015_score_T02', 'KK2015_score_T04',
                              'KK2015_score_T06', 'KK2015_score_T08', 'KK2015_score_T10', 
                              'KK2015_score_T12', 'KK2015_score_T14', 'KK2015_score_T16', 
                              'KK2015_score_T18', 'KK2015_score_T20', 'KK2015_score_T22', 
                              'KK2015_score_T24', 'KK2015_score_T26', 'KK2015_score_28')

KK2015heat_names = ('KK2015_heat_T00', 'KK2015_heat_T02', 'KK2015_heat_T04',
                              'KK2015_heat_T06', 'KK2015_heat_T08', 'KK2015_heat_T10', 
                              'KK2015_heat_T12', 'KK2015_heat_T14', 'KK2015_heat_T16', 
                              'KK2015_heat_T18', 'KK2015_heat_T20', 'KK2015_heat_T22', 
                              'KK2015_heat_T24', 'KK2015_heat_T26', 'KK2015_heat_28')

KK2015cold_names = ('KK2015_cold_T00', 'KK2015_cold_T02', 'KK2015_cold_T04',
                              'KK2015_cold_T06', 'KK2015_cold_T08', 'KK2015_cold_T10', 
                              'KK2015_cold_T12', 'KK2015_cold_T14', 'KK2015_cold_T16', 
                              'KK2015_cold_T18', 'KK2015_cold_T20', 'KK2015_cold_T22', 
                              'KK2015_cold_T24', 'KK2015_cold_T26', 'KK2015_cold_28')

HT2009_names = ('HT2009_log2H/L_asynchronous', 'HT2009_log2H/L_nocodazole', 'HT2009_log2H/L_clb2')

LP2019_names = ('LP2019_akl1_Rep1', 'LP2019_akl1_Rep2', 'LP2019_alk1_Rep1', 'LP2019_alk1_Rep2', 'LP2019_bck1_Rep1',
       'LP2019_bck1_Rep2', 'LP2019_bub1_Rep1', 'LP2019_bub1_Rep2', 'LP2019_bud32_Rep1', 'LP2019_bud32_Rep2',
       'LP2019_cka1_Rep1', 'LP2019_cka1_Rep2', 'LP2019_cka2_Rep1', 'LP2019_cka2_Rep2', 'LP2019_cla4_Rep1',
       'LP2019_cla4_Rep2', 'LP2019_cmk1_Rep1', 'LP2019_cmk1_Rep2', 'LP2019_cmk2_Rep1', 'LP2019_cmk2_Rep2',
       'LP2019_cmp2_Rep1', 'LP2019_cmp2_Rep2', 'LP2019_cna1_Rep1', 'LP2019_cna1_Rep2', 'LP2019_ctk1_Rep1',
       'LP2019_ctk1_Rep2', 'LP2019_dbf2_Rep1', 'LP2019_dbf2_Rep2', 'LP2019_dbf20_Rep1', 'LP2019_dbf20_Rep2',
       'LP2019_dun1_Rep1', 'LP2019_dun1_Rep2', 'LP2019_elm1_Rep1', 'LP2019_elm1_Rep2', 'LP2019_env7_Rep1',
       'LP2019_env7_Rep2', 'LP2019_fpk1_Rep1', 'LP2019_fpk1_Rep2', 'LP2019_fus3_Rep1', 'LP2019_fus3_Rep2',
       'LP2019_hal5_Rep1', 'LP2019_hal5_Rep2', 'LP2019_hog1_Rep1', 'LP2019_hog1_Rep2', 'LP2019_hrk1_Rep1',
       'LP2019_hrk1_Rep2', 'LP2019_hsl1_Rep1', 'LP2019_hsl1_Rep2', 'LP2019_iks1_Rep1', 'LP2019_iks1_Rep2',
       'LP2019_ime2_Rep1', 'LP2019_ime2_Rep2', 'LP2019_isr1_Rep1', 'LP2019_isr1_Rep2', 'LP2019_kcc4_Rep1',
       'LP2019_kcc4_Rep2', 'LP2019_kdx1_Rep1', 'LP2019_kdx1_Rep2', 'LP2019_kin1_Rep1', 'LP2019_kin1_Rep2',
       'LP2019_kin2_Rep1', 'LP2019_kin2_Rep2', 'LP2019_kin4_Rep1', 'LP2019_kin4_Rep2', 'LP2019_kin82_Rep1',
       'LP2019_kin82_Rep2', 'LP2019_kkq8_Rep1', 'LP2019_kkq8_Rep2', 'LP2019_kns1_Rep1', 'LP2019_kns1_Rep2',
       'LP2019_ksp1_Rep1', 'LP2019_ksp1_Rep2', 'LP2019_kss1_Rep1', 'LP2019_kss1_Rep2', 'LP2019_ltp1_Rep1',
       'LP2019_ltp1_Rep2', 'LP2019_mck1_Rep1', 'LP2019_mek1_Rep1', 'LP2019_mek1_Rep2', 'LP2019_mkk1_Rep1',
       'LP2019_mkk1_Rep2', 'LP2019_mkk2_Rep1', 'LP2019_mkk2_Rep2', 'LP2019_mrk1_Rep1', 'LP2019_mrk1_Rep2',
       'LP2019_npr1_Rep1', 'LP2019_npr1_Rep2', 'LP2019_oca1_Rep1', 'LP2019_oca1_Rep2', 'LP2019_pbs2_Rep1',
       'LP2019_pbs2_Rep2', 'LP2019_pkh1_Rep1', 'LP2019_pkh1_Rep2', 'LP2019_pkh2_Rep1', 'LP2019_pkh2_Rep2',
       'LP2019_pkh3_Rep1', 'LP2019_pkh3_Rep2', 'LP2019_pkp1_Rep1', 'LP2019_pkp1_Rep2', 'LP2019_pkp2_Rep1',
       'LP2019_pkp2_Rep2', 'LP2019_pph21_Rep1', 'LP2019_pph21_Rep2', 'LP2019_pph22_Rep1',
       'LP2019_pph22_Rep2', 'LP2019_pph3_Rep1', 'LP2019_pph3_Rep2', 'LP2019_pps1_Rep1', 'LP2019_pps1_Rep2',
       'LP2019_ppt1_Rep1', 'LP2019_ppz1_Rep1', 'LP2019_ppz1_Rep2', 'LP2019_ppz2_Rep1', 'LP2019_ppz2_Rep2',
       'LP2019_prk1_Rep1', 'LP2019_prk1_Rep2', 'LP2019_prr1_Rep1', 'LP2019_prr1_Rep2', 'LP2019_prr2_Rep1',
       'LP2019_prr2_Rep2', 'LP2019_psk2_Rep1', 'LP2019_psk2_Rep2', 'LP2019_psr1_Rep1', 'LP2019_psr1_Rep2',
       'LP2019_psr2_Rep1', 'LP2019_ptc1_Rep1', 'LP2019_ptc1_Rep2', 'LP2019_ptc2_Rep1', 'LP2019_ptc2_Rep2',
       'LP2019_ptc3_Rep1', 'LP2019_ptc3_Rep2', 'LP2019_ptc4_Rep1', 'LP2019_ptc4_Rep2', 'LP2019_ptc5_Rep1',
       'LP2019_ptc5_Rep2', 'LP2019_ptc7_Rep1', 'LP2019_ptk2_Rep1', 'LP2019_ptk2_Rep2', 'LP2019_ptp1_Rep1',
       'LP2019_ptp1_Rep2', 'LP2019_ptp3_Rep1', 'LP2019_ptp3_Rep2', 'LP2019_rck1_Rep1', 'LP2019_rck1_Rep2',
       'LP2019_rck2_Rep1', 'LP2019_rck2_Rep2', 'LP2019_rim11_Rep1', 'LP2019_rim11_Rep2', 'LP2019_rim15_Rep1',
       'LP2019_rim15_Rep2', 'LP2019_rtk1_Rep1', 'LP2019_rtk1_Rep2', 'LP2019_sat4_Rep1', 'LP2019_sat4_Rep2',
       'LP2019_sch9_Rep1', 'LP2019_sch9_Rep2', 'LP2019_scy1_Rep1', 'LP2019_scy1_Rep2', 'LP2019_sdp1_Rep1',
       'LP2019_sdp1_Rep2', 'LP2019_sit4_Rep1', 'LP2019_sit4_Rep2', 'LP2019_siw14_Rep1', 'LP2019_siw14_Rep2',
       'LP2019_skm1_Rep1', 'LP2019_skm1_Rep2', 'LP2019_sks1_Rep1', 'LP2019_sks1_Rep2', 'LP2019_sky1_Rep1',
       'LP2019_sky1_Rep2', 'LP2019_slt2_Rep1', 'LP2019_smk1_Rep1', 'LP2019_smk1_Rep2', 'LP2019_ssk2_Rep1',
       'LP2019_ssk2_Rep2', 'LP2019_ssk22_Rep1', 'LP2019_ssk22_Rep2', 'LP2019_ste11_Rep1',
       'LP2019_ste11_Rep2', 'LP2019_ste7_Rep1', 'LP2019_ste7_Rep2', 'LP2019_swe1_Rep1', 'LP2019_swe1_Rep2',
       'LP2019_tda1_Rep1', 'LP2019_tda1_Rep2', 'LP2019_tel1_Rep1', 'LP2019_tel1_Rep2', 'LP2019_tor1_Rep1',
       'LP2019_tor1_Rep2', 'LP2019_tos3_Rep1', 'LP2019_tos3_Rep2', 'LP2019_tpk2_Rep1', 'LP2019_tpk2_Rep2',
       'LP2019_tpk3_Rep1', 'LP2019_tpk3_Rep2', 'LP2019_vhs1_Rep1', 'LP2019_vhs1_Rep2', 'LP2019_ych1_Rep1',
       'LP2019_ych1_Rep2', 'LP2019_yck1_Rep1', 'LP2019_yck1_Rep2', 'LP2019_yck2_Rep1', 'LP2019_yck2_Rep2', 
       'LP2019_ygk3_Rep1', 'LP2019_ygk3_Rep2', 'LP2019_ypk1_Rep1', 'LP2019_ypk1_Rep2', 'LP2019_ypk2_Rep1', 
       'LP2019_ypk2_Rep2', 'LP2019_ypk3_Rep1', 'LP2019_ypk3_Rep2', 'LP2019_ypl150w_Rep1', 'LP2019_ypl150w_Rep2',
       'LP2019_yvh1_Rep1', 'LP2019_yvh1_Rep2')

ZW2019_names = ('ZW2019_126_rep1', 'ZW2019_127N_rep1', 'ZW2019_127C_rep1', 'ZW2019_128N_rep1',
                'ZW2019_128C_rep1', 'ZW2019_129N_rep1', 'ZW2019_129C_rep1', 'ZW2019_130N_rep1',
                'ZW2019_130C_rep1', 'ZW2019_131_rep1', 'ZW2019_126_rep2', 'ZW2019_127N_rep2', 
                'ZW2019_127C_rep2', 'ZW2019_128N_rep2', 'ZW2019_128C_rep2', 'ZW2019_129N_rep2', 
                'ZW2019_129C_rep2', 'ZW2019_130N_rep2', 'ZW2019_130C_rep2', 'ZW2019_131_rep2')

GP2021_names = ('GP2021_126_rep1', 'GP2021_127n_rep1', 'GP2021_127c_rep1', 'GP2021_128n_rep1',
                'GP2021_128c_rep1', 'GP2021_129n_rep1', 'GP2021_129c_rep1', 'GP2021_130n_rep1',
                'GP2021_130c_rep1', 'GP2021_131_rep1', 'GP2021_131c_rep1', 'GP2021_126_rep2', 
                'GP2021_127n_rep2', 'GP2021_127c_rep2', 'GP2021_128n_rep2', 'GP2021_128c_rep2', 
                'GP2021_129n_rep2', 'GP2021_129c_rep2', 'GP2021_130n_rep2', 'GP2021_130c_rep2', 
                'GP2021_131_rep2', 'GP2021_131c_rep2', 'GP2021_126_rep3', 'GP2021_127n_rep3', 
                'GP2021_127c_rep3', 'GP2021_128n_rep3', 'GP2021_128c_rep3', 'GP2021_129n_rep3', 
                'GP2021_129c_rep3', 'GP2021_130n_rep3', 'GP2021_130c_rep3', 'GP2021_131_rep3',
                'GP2021_131c_rep3')

CP2024a_names = ('CP2024a_50min', 'CP2024a_60min', 'CP2024a_70min', 'CP2024a_80min', 'CP2024a_90min',
                  'CP2024a_100min', 'CP2024a_110min', 'CP2024a_120min', 'CP2024a_130min', 'CP2024a_140min')

CP2024b_names = ('CP2024b_50min', 'CP2024b_60min', 'CP2024b_70min', 'CP2024b_80min', 'CP2024b_90min',
                  'CP2024b_100min', 'CP2024b_110min', 'CP2024b_120min', 'CP2024b_130min', 'CP2024b_140min')

CP2024c_names = ('CP2024c_60min_control', 'CP2024c_65min_control', 'CP2024c_70min_control', 
                  'CP2024c_75min_control', 'CP2024c_80min_control', 'CP2024c_85min_control', 
                  'CP2024c_65min_3MBPP1', 'CP2024c_70min_3MBPP1', 'CP2024c_75min_3MBPP1', 
                  'CP2024c_80min_3MBPP1', 'CP2024c_85min_3MBPP1')

CP2024d_names = ('CP2024d_60min_control', 'CP2024d_65min_control', 'CP2024d_70min_control', 
                  'CP2024d_75min_control', 'CP2024d_80min_control', 'CP2024d_85min_control', 
                  'CP2024d_65min_3MBPP1', 'CP2024d_70min_3MBPP1', 'CP2024d_75min_3MBPP1', 
                  'CP2024d_80min_3MBPP1', 'CP2024d_85min_3MBPP1')

CP2024e_names = ('CP2024e_60min_control', 'CP2024e_65min_control', 'CP2024e_70min_control', 
                  'CP2024e_75min_control', 'CP2024e_80min_control', 'CP2024e_85min_control', 
                  'CP2024e_65min_3MBPP1', 'CP2024e_70min_3MBPP1', 'CP2024e_75min_3MBPP1', 
                  'CP2024e_80min_3MBPP1', 'CP2024e_85min_3MBPP1')

# KS2024a_names = ('KS2024a_intensity0_rep1', 'KS2024a_intensity0_rep2', 'KS2024a_intensity0_rep3', 
#                   'KS2024a_intensity1_rep1', 'KS2024a_intensity1_rep2', 'KS2024a_intensity1_rep3', 
#                   'KS2024a_intensity2_rep1', 'KS2024a_intensity2_rep2', 'KS2024a_intensity2_rep3',
#                   'KS2024a_intensity3_rep1', 'KS2024a_intensity3_rep2', 'KS2024a_intensity3_rep3', 
#                   'KS2024a_intensity4_rep1', 'KS2024a_intensity4_rep2', 'KS2024a_intensity4_rep3',
#                   'KS2024a_intensity5_rep1', 'KS2024a_intensity5_rep2', 'KS2024a_intensity5_rep3',
#                   'KS2024a_intensity6_rep1', 'KS2024a_intensity6_rep2', 'KS2024a_intensity6_rep3',
#                   'KS2024a_intensity7_rep1', 'KS2024a_intensity7_rep2', 'KS2024a_intensity7_rep3',
#                   'KS2024a_intensity8_rep1', 'KS2024a_intensity8_rep2', 'KS2024a_intensity8_rep3',
#                   'KS2024a_intensity9_rep1', 'KS2024a_intensity9_rep2', 'KS2024a_intensity9_rep3')

# KS2024b_names = ('KS2024b_intensity0_rep1', 'KS2024b_intensity0_rep2', 'KS2024b_intensity0_rep3', 
#                   'KS2024b_intensity1_rep1', 'KS2024b_intensity1_rep2', 'KS2024b_intensity1_rep3', 
#                   'KS2024b_intensity2_rep1', 'KS2024b_intensity2_rep2', 'KS2024b_intensity2_rep3',
#                   'KS2024b_intensity3_rep1', 'KS2024b_intensity3_rep2', 'KS2024b_intensity3_rep3', 
#                   'KS2024b_intensity4_rep1', 'KS2024b_intensity4_rep2', 'KS2024b_intensity4_rep3',
#                   'KS2024b_intensity5_rep1', 'KS2024b_intensity5_rep2', 'KS2024b_intensity5_rep3',
#                   'KS2024b_intensity6_rep1', 'KS2024b_intensity6_rep2', 'KS2024b_intensity6_rep3',
#                   'KS2024b_intensity7_rep1', 'KS2024b_intensity7_rep2', 'KS2024b_intensity7_rep3',
#                   'KS2024b_intensity8_rep1', 'KS2024b_intensity8_rep2', 'KS2024b_intensity8_rep3',
#                   'KS2024b_intensity9_rep1', 'KS2024b_intensity9_rep2', 'KS2024b_intensity9_rep3')

PC2024a_names = ('PC2024a_D0_1', 'PC2024a_D0_2', 'PC2024a_D0_3', 
                 'PC2024a_D2.5_1', 'PC2024a_D2.5_2', 
                  'PC2024a_D2.5_3', 'PC2024a_D5_1', 'PC2024a_D5_2', 'PC2024a_D5_3', 'PC2024a_D10_1', 
                  'PC2024a_D10_2', 'PC2024a_D10_3', 'PC2024a_D120_1', 'PC2024a_D120_2', 'PC2024a_D120_3', 
                  'PC2024a_N0_1', 'PC2024a_N0_2', 'PC2024a_N0_3', 'PC2024a_N2.5_1', 'PC2024a_N2.5_2', 
                  'PC2024a_N2.5_3', 'PC2024a_N5_1', 'PC2024a_N5_2', 'PC2024a_N5_3', 'PC2024a_N10_1', 
                  'PC2024a_N10_2','PC2024a_N10_3', 'PC2024a_N120_1', 'PC2024a_N120_2', 'PC2024a_N120_3')

PC2024b_names = ('PC2024b_N5_1', 'PC2024b_N5_2', 'PC2024b_N10_1', 'PC2024b_N10_2', 'PC2024b_0_1',
                  'PC2024b_0_2', 'PC2024b_5_1', 'PC2024b_5_2', 'PC2024b_10_2', 'PC2024b_20_1', 
                  'PC2024b_20_2', 'PC2024b_40_1', 'PC2024b_40_2', 'PC2024b_70_1', 'PC2024b_70_2', 
                  'PC2024b_120_1', 'PC2024b_120_2')

PC2024c_names = ('PC2024c_N2.5_1', 'PC2024c_N2.5_2', 'PC2024c_N5_1', 'PC2024c_N5_2', 
                  'PC2024c_D0_1', 'PC2024c_D0_2','PC2024c_D0.5_1',  'PC2024c_D0.5_2',
                  'PC2024c_D1_1', 'PC2024c_D1_2', 'PC2024c_D1.5_1', 'PC2024c_D1.5_2', 
                  'PC2024c_D2_1', 'PC2024c_D2_2', 'PC2024c_D2.5_1', 'PC2024c_D2.5_2', 
                  'PC2024c_D5_1', 'PC2024c_D5_2')

PC2024d_names = ('PC2024d_0_1', 'PC2024d_1_1', 'PC2024d_10_1', 'PC2024d_50_1', 
                  'PC2024d_100_1', 'PC2024d_500_1','PC2024d_1000_1',  'PC2024d_1_2',
                  'PC2024d_10_2', 'PC2024d_50_2', 'PC2024d_100_2', 'PC2024d_500_2', 
                  'PC2024d_0_2')

PC2024e_names = ('PC2024e_SGglc_10_1', 'PC2024e_SGglc_60_1', 'PC2024e_SGglc_1_1', 'PC2024e_SGglc_15_1', 
                  'PC2024e_SGglc_5_2', 'PC2024e_SGglc_2.5_1', 'PC2024e_SGglc_15_2', 'PC2024e_SGglc_30_1', 
                  'PC2024e_SGglc_1_2', 'PC2024e_SGglc_10_2', 'PC2024e_befGlc_1', 'PC2024e_SGglc_30_2', 
                  'PC2024e_befGlc_2', 'PC2024e_SGglc_2.5_2', 'PC2024e_SGglc_0_1', 'PC2024e_SGglc_5_1', 
                  'PC2024e_SGglc_60_2', 'PC2024e_SGglc_0_2', 'PC2024e_ss_0_2', 'PC2024e_ss_2.5_2', 
                  'PC2024e_ss_1_1', 'PC2024e_ss_1_2', 'PC2024e_ss_60_2', 'PC2024e_ss_10_1', 'PC2024e_ss_5_1',
                  'PC2024e_ss_15_1', 'PC2024e_ss_0_1', 'PC2024e_ss_5_2', 'PC2024e_ss_30_2', 
                  'PC2024e_ss_60_1', 'PC2024e_ss_2.5_1', 'PC2024e_ss_30_1', 'PC2024e_ss_10_2', 'PC2024e_ss_15_2')	

PC2024f_names = ('PC2024f_none_D0_1', 'PC2024f_none_N0_1', 'PC2024f_none_N0_2', 'PC2024f_none_D5_1',	
                  'PC2024f_none_N5_1', 'PC2024f_none_N5_2', 'PC2024f_Tpk2_D0_1', 'PC2024f_Tpk2_N0_1',	
                  'PC2024f_Tpk2_N0_2', 'PC2024f_Tpk2_D5_1', 'PC2024f_Tpk2_N5_1', 'PC2024f_Tpk2_N5_2',	
                  'PC2024f_Tpk1_D0_1', 'PC2024f_Tpk1_N0_1', 'PC2024f_Tpk1_N0_2', 'PC2024f_Tpk1_D5_1',	
                  'PC2024f_Tpk1_N5_1', 'PC2024f_Tpk1_N5_2', 'PC2024f_Tpk3_N0_1', 'PC2024f_Tpk3_N0_2',	
                  'PC2024f_Tpk3_D5_1', 'PC2024f_Tpk3_N5_1', 'PC2024f_Tpk3_N5_2')	

PC2024g_names = ('PC2024g_A_0N_0', 'PC2024g_B_0N_0', 'PC2024g_C_0N_0', 'PC2024g_D_0N_0', 
                  'PC2024g_C_500N_0', 'PC2024g_D_500N_0', 'PC2024g_A_0N_5', 'PC2024g_B_0N_5', 
                  'PC2024g_C_0N_5', 'PC2024g_A_2.5N_5', 'PC2024g_B_2.5N_5', 'PC2024g_A_5N_5', 
                  'PC2024g_B_5N_5', 'PC2024g_A_10N_5', 'PC2024g_B_10N_5', 'PC2024g_A_20N_5', 
                  'PC2024g_B_20N_5', 'PC2024g_A_40N_5', 'PC2024g_B_40N_5', 'PC2024g_A_80N_5', 
                  'PC2024g_B_80N_5', 'PC2024g_A_160N_5', 'PC2024g_B_160N_5', 'PC2024g_B_500N_5',
                  'PC2024g_C_500N_5', 'PC2024g_D_500N_5', 'PC2024g_B_500N_120', 'PC2024g_C_500N_120',
                  'PC2024g_D_500N_120')

PC2024h_names = ('PC2024h_N2', 'PC2024h_N3', 'PC2024h_N4', 'PC2024h_D1', 
                  'PC2024h_D2', 'PC2024h_D3', 'PC2024h_D4')

PC2024i_names = ('PC2024i_D2', 'PC2024i_D4', 'PC2024i_D6', 'PC2024i_D8', 
                  'PC2024i_N1', 'PC2024i_N3', 'PC2024i_N5', 'PC2024i_N7')

DS2021_names = ('DS2021_1', 'DS2021_2', 'DS2021_3', 'DS2021_4', 'DS2021_5',
                  'DS2021_6', 'DS2021_7', 'DS2021_8', 'DS2021_9', 'DS2021_10',
                  'DS2021_11', 'DS2021_12', 'DS2021_13', 'DS2021_14', 'DS2021_15', 
                  'DS2021_16', 'DS2021_17', 'DS2021_18', 'DS2021_19', 'DS2021_20', 
                  'DS2021_21', 'DS2021_22', 'DS2021_23', 'DS2021_24', 'DS2021_25', 
                  'DS2021_26', 'DS2021_27', 'DS2021_28', 'DS2021_29', 'DS2021_30', 
                  'DS2021_31', 'DS2021_32', 'DS2021_33', 'DS2021_34', 'DS2021_35', 
                  'DS2021_36', 'DS2021_37', 'DS2021_38', 'DS2021_39', 'DS2021_40', 
                  'DS2021_41', 'DS2021_42', 'DS2021_43', 'DS2021_44', 'DS2021_45', 
                  'DS2021_46', 'DS2021_47', 'DS2021_48', 'DS2021_49', 'DS2021_50')

SW2021_names = ('SW2021')

VA2020_names = ('VA2020_P1_T0', 'VA2020_P1_T30', 'VA2020_P1_T60', 'VA2020_P1_T120', 'VA2020_P1_T240', 
                'VA2020_P2_T0', 'VA2020_P2_T30', 'VA2020_P2_T60', 'VA2020_P2_T120', 'VA2020_P2_T240',
                'VA2020_P3_T0', 'VA2020_P3_T30', 'VA2020_P3_T60', 'VA2020_P3_T120', 'VA2020_P3_T240',
                'VA2020_P4_T0', 'VA2020_P4_T30', 'VA2020_P4_T60', 'VA2020_P4_T120', 'VA2020_P4_T240')

SC2021a_names = ('SC2021a_1', 'SC2021a_2')

SC2021b_names = ('SC2021b_1', 'SC2021b_2')

SF2023_names = ('SF2023')

GP2024_names = ('GP2024')

EC2021_names = ('EC2021_con0', 'EC2021_con10', 'EC2021_con20', 'EC2021_con30', 'EC2021_con40', 'EC2021_con50', 
                'EC2021_con60', 'EC2021_con70', 'EC2021_con80', 'EC2021_con90', 'EC2021_0', 'EC2021_10',
                'EC2021_20', 'EC2021_30', 'EC2021_40', 'EC2021_50', 'EC2021_60', 'EC2021_70', 'EC2021_80', 'EC2021_90')

LT2024_names = ('LT2024_0', 'LT2024_30', 'LT2024_45', 'LT2024_60', 'LT2024_75', 'LT2024_90',
                'LT2024_105', 'LT2024_120', 'LT2024_135', 'LT2024_150')



all_names = (MC2020_names + TK2018_names + RP2022_names + 
             RP2023_names + LY2021_names + VB2014_names + 
             KK2015score_names + KK2015heat_names + KK2015cold_names + 
             HT2009_names + LP2019_names + ZW2019_names + 
             GP2021_names + CP2024a_names + CP2024b_names + 
             CP2024c_names + CP2024d_names + CP2024e_names +
            #  KS2024a_names + KS2024b_names + 
             PC2024a_names + PC2024b_names + PC2024c_names + PC2024d_names + 
             PC2024e_names + PC2024f_names + PC2024g_names + PC2024h_names + 
             PC2024i_names + DS2021_names +
             (SW2021_names,) + VA2020_names + SC2021a_names + 
             SC2021b_names + (SF2023_names,) + (GP2024_names,) +
             EC2021_names + LT2024_names)


# ----------------- #
# CREATE DICTIONARY OF DATAFRAMES
# ----------------- #

dataset_dict = {}

# add each dataframe from list to dictionary (with index)
for i, dataset_name in enumerate(all_names):
    dataset_dict[dataset_name] = dataset_list[i] 
    
print(dataset_dict['MC2020_6h_rep1'].head())


# ----------------- #
# CREATE MATRIX HEADER
# ----------------- #

norm_matrix = normalising.MinMax_normalize_and_merge(dataset_dict, MinMaxScaler())

norm_matrix.columns.name = ''

# get unique phosphosites
uniq_phos = norm_matrix['phosphosite_ID'].unique()

if len(uniq_phos) == len(norm_matrix):
    print('Passed! Number of unique phosphosites matches length of merged dataframe')
else:
    print('Failed! Number of unique phosphosites does not match length of merged dataframe')

matrix_cols = pd.DataFrame(columns = uniq_phos)

matrix_cols.to_csv('/data/home/bty449/ExplainableAI/Normalisation/normalised-matrix-header.csv', index = False)

print('Matrix header saved to CSV.', matrix_cols)


# ----------------- #
# FORMAT MATRIX
# ----------------- #

transposed_matrix = norm_matrix.T
print(f'Transposed matrix:', transposed_matrix.head())

# set first row to column header
transposed_matrix.columns = transposed_matrix.iloc[0]

# remove first row
transposed_matrix = transposed_matrix[1:]

# rename and reset column index
transposed_matrix = transposed_matrix.rename_axis('DatasetName').reset_index()
print(f'Transposed matrix:', transposed_matrix.head())


cols = [i for i in transposed_matrix.columns if i not in ['DatasetName']]

for col in cols:
    transposed_matrix[col] = pd.to_numeric(transposed_matrix[col])
    
minmax_matrix = transposed_matrix.replace([np.inf, -np.inf], np.nan)
print(f'MinMax matrix:', minmax_matrix.head())

minmax_matrix.to_csv('/data/home/bty449/ExplainableAI/MatrixCSVs/NormalisedMatrix.csv', index = False)
print(f'Normalised matrix saved successfully!', minmax_matrix)