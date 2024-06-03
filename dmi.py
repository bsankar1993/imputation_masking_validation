import pandas as pd
import numpy as np
import os
import sys 
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer


class ImputeAndCalculateRMSE:
    def __init__(self, n_masked, exp_type):
        self.n_masked = n_masked
        self.filepath_common = 'proturn_output_'
        self.strains = ['aj', 'balbc', 'c57', 'cej', 'dba', 'fvb']
        self.exp_type = exp_type
        self.mask = f'hl-data_masked_{n_masked}.csv'
        self.true = 'hl-data_filtered.csv'
        self.imputer_mean = SimpleImputer(strategy='mean')
        self.output_dir = 'rmse'
        self.imputed_dir = 'dmi_imputed_data'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.imputed_dir):
            os.makedirs(self.imputed_dir)

    def get_file_paths(self):
        file_paths_masked = [f'{self.filepath_common}{s}_{self.exp_type}/{self.mask}' for s in self.strains]
        file_paths_true = [f'{self.filepath_common}{s}_{self.exp_type}/{self.true}' for s in self.strains]
        return file_paths_masked, file_paths_true
    
    # Function to perform multiple imputation on a peptide in the dataset and return the average squared error across the 10 iterations 
    def peptide_dmi_and_rmse(self, data, true, file_path, n_imputations=10):
        se_i = []
        for i in range(n_imputations):
            random_state = np.random.randint(0, 10000)
            imputer = IterativeImputer(sample_posterior=True, random_state=random_state)
            imputed_data = imputer.fit_transform(data)
            imputed_vals = imputed_data[:,0]
            true_vals = true['A0'].values
            se_i.append(np.sum((imputed_vals-true_vals)**2))

            # Convert back to DataFrame for easier handling
            imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
            
            # Save the imputed DataFrame to a CSV file
            imputed_df.to_csv(os.path.join(output_dir, f'imputed_data_{i+1}.csv'), index=False)
            #print(f'Imputation {i+1} completed and saved with random state {random_state}.')
        se_i = sum(se_i)/n_imputations
        return se_i

    def calculate_rmse(self, file_paths_masked, file_paths_true):
        n_ctrl = []
        mse_sum_strain = []

        for mask, un_masked in zip(file_paths_masked, file_paths_true):
            try:
                data_mask = pd.read_csv(mask)
                #filter the dataframe to only include columns ID, t and A0
                data_mask = data_mask[['ID', 't', 'A0']]
        
                data_true = pd.read_csv(un_masked)
                #filter the dataframe to only include columns ID, t and A0
                data_true = data_true[['ID', 't', 'A0']]
                
            except FileNotFoundError:
                print(f"File not found: {mask} or {un_masked}")
                continue
            except Exception as e:
                print(f"An error occurred while reading {mask} or {un_masked}: {e}")
                continue
            
            n_ctrl.append(len(data_mask['A0']))
            mse_k = []
            print(f'imputing for {mask}...\n')
            for ID in data_mask['ID'].unique():
                idx = data_mask.loc[data_mask['ID'] == ID].index
                #get t and A0 vales for the idx values
                peptide_dat = data_mask.loc[idx, ['t', 'A0']]
                true_peptide_dat = data_true.loc[idx, ['t','A0']] 
                se = self.peptide_dmi_and_rmse(peptide_dat, true_peptide_dat, mask, n_imputations=10)
                #squared error for the imputed values for each peptide, these will be summed up later            
                mse_k.append(se)
                
            #sum all the squared errors for each peptide in the strain 
            mse_sum_strain.append(sum(mse_k))
            
        #sum all the squared errors for each strain in the experiment group 
        mse_sum_ctrl = sum(mse_sum_strain)
        n_ctrl_total = sum(n_ctrl) - 1 # -1 to account for the degrees of freedom
        #calculate the root mean squared error for the experiment group
        rmse_ctrl = np.sqrt(mse_sum_ctrl / n_ctrl_total)
        return rmse_ctrl

    def save_rmse(self, rmse_value):
        with open(f'./{self.output_dir}/rmse_{self.exp_type}.csv', 'a') as f:
            f.write(f'{rmse_value}, {self.n_masked}, mean_single_impute\n') 

def main(n_masked, exp_type):
    imputer = ImputeAndCalculateRMSE(n_masked, exp_type)
    file_paths_masked, file_paths_true = imputer.get_file_paths()
    rmse_value = imputer.calculate_rmse(file_paths_masked, file_paths_true)
    imputer.save_rmse(rmse_value)
    print(f"RMSE for n_masked={n_masked}, exp_type={exp_type}: {rmse_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mean imputation and calculate RMSE.')
    parser.add_argument('n_masked', type=int, choices=[1, 2, 3, 4], help='Number of masked values')
    parser.add_argument('exp_type', type=str, choices=['ctrl', 'iso'], help='Experiment type')
    args = parser.parse_args()
    main(args.n_masked, args.exp_type)