import os
import pandas as pd
import numpy as np
import argparse
import json

class ImputationAnalysis:
    def __init__(self, imputation, n_imputations):
        self.imputation = imputation
        self.n_imputations = n_imputations
        self.filepath_common = 'proturn_output_'
        self.strains = ['aj', 'balbc', 'c57', 'cej', 'dba', 'fvb']
        self.exp = ['ctrl', 'iso']
        self.true_data_files = 'hl-data_filtered.csv'
        self.mask = [1, 2, 3, 4, 5]
        self.masked_data_files = 'hl-data_masked_'
        self.imputed_data_files = f'{imputation}'
        self.output_dir = 'nrmse'
        self.folders = [self.filepath_common + s + '_' + e + '/' for s in self.strains for e in self.exp]
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def run_analysis(self):
        scores = {}
        for m in self.mask:
            sse_mask = []
            ovss = []
            scores[m] = {}
            for f in self.folders:
                masked_data = pd.read_csv(f + self.masked_data_files + str(m) + '.csv')
                masked_indices = masked_data.loc[masked_data['A0'].isnull(), 'A0'].index
                
                imputed_data = pd.read_csv(f + self.imputed_data_files + "_" + str(m) + '.csv')
                true_data = pd.read_csv(f + self.true_data_files)
                ovss.append(np.sum(true_data.loc[masked_indices, 'A0'] ** 2))
                
                true_data = true_data.pivot(index='ID', columns='t', values='A0').reset_index()
                
                sse_n_total = []
                
                for i in range(self.n_imputations):
                    imputed_subset = imputed_data.copy()
                    
                    if self.imputation == 'dmi_np':
                        imputed_subset = imputed_subset[imputed_subset['.imp'] == i + 1]
                    else: 
                        pass 
                    
                    true = np.array(true_data.drop(columns=['ID']))
                    
                    #if dmi imputation 
                    if self.imputation == 'dmi_np':
                        imputed = np.array(imputed_subset.drop(columns=['.id', '.imp', 'Unnamed: 0']))
                    #if si_mean or knn
                    elif self.imputation == 'si_mean' or self.imputation == 'knn' or self.imputation == 'knn_30':
                        imputed = np.array(imputed_subset.drop(columns=['Unnamed: 0']))
                    
                    sse_n = np.sum((true - imputed) ** 2)
                    sse_n_total.append(sse_n)

                sse_n_total = np.mean(sse_n_total)
                sse_mask.append(sse_n_total)
            
            
            nrmse = np.sum(sse_mask) / sum(ovss)
            print(f'{nrmse} for {m}')
            scores[m]['nrmse'] = nrmse
            scores[m]['ovss'] = ovss
            scores[m]['sse_mask'] = sse_mask
            scores[m]['nrmse_individual'] = [sse / ovss[i] for i, sse in enumerate(sse_mask)]


            
            with open(f'{self.output_dir}/{self.imputation}_nrmse_a0.json', 'w') as f:
                json.dump(scores, f, indent=2)

def main():
    # parser = argparse.ArgumentParser(description='Run imputation analysis.')
    # parser.add_argument('imputation', type=str, help='Type of imputation method: {dmi_np, si_mean, knn_30}')
    # parser.add_argument('n_imputations', type=int, help='Number of imputated datasets; 10 if dmi_np, 1 otherwise')
    # args = parser.parse_args()
    
    analysis = ImputationAnalysis('dmi_np', 10)
    analysis.run_analysis()
    analysis = ImputationAnalysis('si_mean', 1)
    analysis.run_analysis()
    analysis = ImputationAnalysis('knn_30', 1)
    analysis.run_analysis()

if __name__ == "__main__":
    main()
