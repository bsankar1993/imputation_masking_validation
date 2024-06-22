import os
import pandas as pd
import numpy as np
import argparse
import pdb
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ImputationAnalysis:
    def __init__(self, imputation):
        self.imputation = imputation
        self.filepath_common = 'proturn_output_'
        self.strains = ['aj', 'balbc', 'c57', 'cej', 'dba', 'fvb']
        self.exp = ['ctrl', 'iso']
        self.true_data_files = 'hl.out'
        self.mask = [1, 2, 3, 4, 5]
        self.output_dir = 'nrmse_k'
        self.folders = [self.filepath_common + s + '_' + e + '/' for s in self.strains for e in self.exp]

        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_nrmse(self, imputed_data, true_data):
        imputed_data = imputed_data.loc[(imputed_data['k'] >= 0) & (imputed_data['k'] <= 60)] 
        true_data = true_data.loc[(true_data['k'] >= 0) & (true_data['k'] <= 60)] 
        true_data = true_data.loc[(true_data['R2'] >= 0.8) | (true_data['SS'] <= 0.05)] 

        sse = 0 
        nf = 0 
        count = 0

        trues = []
        imputes = []

        for _id in imputed_data['ID'].unique():
            if self.imputation == 'imp': 
                #imputed data for a specific protein
                imp_prot = imputed_data.loc[imputed_data['ID'] == _id]#.groupby(['ID', 'Uniprot']).agg({'k': 'mean'}).reset_index()
                #true data for a specific protein
                t = true_data.loc[true_data['ID'] == _id]
                # pdb.set_trace()
                # filter t so it only contains rows where ID is in imp['ID']
                # t = t.loc[t['ID'].isin(imp_prot['ID'])]
                if len(t) == 0:
                    continue
                error = np.median(t['k']) - np.median(imp_prot['k'])
                sse += error**2
                nf += np.median(t['k'])**2
                count+=1
            else: 
                #imputed data for a specific protein
                
                imp_prot = imputed_data.loc[imputed_data['ID'] == _id]
                #true data for a specific protein
                t = true_data.loc[true_data['ID'] == _id]
                # filter t so it only contains rows where ID is in imp['ID']
                # t = t.loc[t['ID'].isin(imp_prot['ID'])]
                if len(t) == 0:
                    continue
                error = np.median(t['k']) - np.median(imp_prot['k'])
                sse += error**2
                nf += np.median(t['k'])**2
                count+=1
            trues.append(np.median(t['k']))
            imputes.append(np.median(imp_prot['k']))
        return sse, nf, count, trues, imputes

    def run_analysis(self):
        scores = {}
        sse_mask = {}
        nf_mask = {}
        for m in self.mask:
            scores[m] = {}
            sse_mask[m] = []
            nf_mask[m] = []
            counts = []
            trues_all = []
            imputes_all = []
            for f in self.folders:
                try:
                    # Read in file
                    imputed_data = pd.read_csv(f'{f}ProTurn_{self.imputation}_{m}.txt', sep='\t')
                    true_data = pd.read_csv(f'{f}hl.out', sep='\t')
                except FileNotFoundError:
                    print(f"File not found: {f}")
                    continue

                sse,nf, count, trues, imputes = self.calculate_nrmse(imputed_data, true_data) 

                sse_mask[m].append(sse)
                nf_mask[m].append(nf)
                counts.append(count)
                trues_all.extend(trues)
                imputes_all.extend(imputes)

                # plot scatter plot of trues versus imputes
            temp_dat = pd.DataFrame({'trues':trues_all, 'imputes':imputes_all})
            sns.scatterplot(temp_dat, x='trues', y='imputes')
            plt.savefig(os.path.join(self.output_dir, f'scatter_{self.imputation}_{m}.png'))
            plt.close()

            sse_total = sum(sse_mask[m])
            nf_total = sum(nf_mask[m])
            nrmse = sse_total/nf_total
            print(f'{nrmse} for {m}')

            scores[m]['nrmse'] = nrmse
            scores[m]['sse'] = sse_total
            scores[m]['nf'] = nf_total
            scores[m]['sse_individual'] = sse_mask[m]
            scores[m]['nf_individual'] = nf_mask[m]
            scores[m]['nrmses'] = [sse/nf for sse, nf in zip(sse_mask[m], nf_mask[m])]
            scores[m]['count'] = sum(counts)
        with open(f'{self.output_dir}/{self.imputation}_nrmse.json', 'w') as f:
            json.dump(scores, f, indent=2)

def plot_comparison():
    imps = ['knn','imp','si']
    ms = list(range(1,6))
    dats = {imp:json.load(open(f'nrmse_k/{imp}_nrmse.json','r')) for imp in imps}
    
    # plot nrmse for each imputation method over m in one figure
    for imp in imps:
        nrmse = [dats[imp][str(m)]['nrmse'] for m in ms]
        plt.plot(ms,nrmse,label=imp)

    plt.xlabel('m')
    plt.ylabel('nrmse')
    plt.legend()
    plt.savefig('nrmse_k/nrmse_m.pdf', dpi=300)
    plt.close()

    # plot sse normalized by count for each imputation method over m in one figure
    for imp in imps:
        sse = [dats[imp][str(m)]['sse']/dats[imp][str(m)]['count'] for m in ms]
        plt.plot(ms,sse,label=imp)
    plt.xlabel('m')
    plt.ylabel('sse')
    plt.legend()
    plt.savefig('nrmse_k/sse_m.pdf', dpi=300)
    plt.close()



def main():
    # parser = argparse.ArgumentParser(description='Run imputation analysis.')
    # parser.add_argument('imputation', type=str, choices=['imp', 'si', 'knn'], help='Type of imputation method: {imp, si, knn}')
    # args = parser.parse_args()

    # analysis = ImputationAnalysis(args.imputation)
    analysis = ImputationAnalysis('imp')
    analysis.run_analysis()
    analysis = ImputationAnalysis('si')
    analysis.run_analysis()
    analysis = ImputationAnalysis('knn')
    analysis.run_analysis()

    plot_comparison()


if __name__ == "__main__":
    main()
