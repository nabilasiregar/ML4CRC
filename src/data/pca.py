import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PCAAnalysis:
    def __init__(self, file_path, reduced_data_path):
        self.file_path = file_path
        self.reduced_data_path = reduced_data_path

    def perform_pca(self):
        df = pd.read_csv(self.file_path)
        df_for_pca = df.iloc[:, 1:]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_for_pca)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
        pca_df.to_csv(self.reduced_data_path, index=False)
        self.plot_pca_results(pca_df)

    @staticmethod
    def plot_pca_results(pca_df):
        plt.figure(figsize=(8,6))
        plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA on Gene Count Data')
        plt.show()
