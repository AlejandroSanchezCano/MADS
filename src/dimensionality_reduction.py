# Built-in libraries

# Third-party libraries
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Custom libraries


class DimensionalityReduction:
    '''
    Dimensionality reduction techniques to visualize the embeddings.
    '''
    def __init__(self, embedding: pd.DataFrame, draw: pd.Series):
        self.X = embedding
        self.draw = draw.loc[embedding.index]

    def pca(self):
        # Run PCA
        n_components = int(self.X.shape[1] ** 0.5)
        pca = PCA(n_components = n_components)
        pca_result = pca.fit_transform(self.X)

        # Build PCA data frame
        columns = [f'PC{i + 1} ({var:.2f})%' for i, var in enumerate(pca.explained_variance_ratio_ * 100)]
        df = pd.DataFrame(pca_result, columns = columns, index = self.X.index)
        
        # Plot PCA
        sns.scatterplot(data = df, x = columns[0], y = columns[1])
        plt.savefig('pca.png')

        # Scree plot
        plt.figure()
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance %')
        plt.title('Scree Plot')
        plt.savefig('scree_plot.png')        

    def tsne(self):
        # Run t-SNE
        n_components = 2
        tsne = TSNE(n_components = n_components, perplexity = 30, random_state = 42)
        tsne_result = tsne.fit_transform(self.X)
        
        # Build t-SNE data frame
        columns = ['x t-SNE', 'y t-SNE'] + ['t-SNE'] * (n_components - 2)
        df = pd.DataFrame(tsne_result, columns = columns, index = self.X.index)

        # Plot t-SNE
        sns.scatterplot(data = df, x = columns[0], y = columns[1], hue = self.draw)
        plt.savefig('tsne.png')

    def umap(self):
        # Run UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(self.X)
        df = pd.DataFrame(embedding, columns = ['x UMAP', 'y UMAP'], index = self.X.index)

        # Plot UMAP
        sns.scatterplot(data = df, x = 'x UMAP', y = 'y UMAP', hue = self.draw)
        plt.savefig('umap.png')
        
if __name__ == '__main__':
    from esm2_experiments import ESM2Experiment
    esm2 = ESM2Experiment()
    df = esm2.build_embedding(type = 'per_protein', species = 'all')
    species = esm2.species()
    print(len(species.unique()))
    dr = DimensionalityReduction(embedding = df, draw = species)
    dr.umap()