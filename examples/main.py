import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from crispyn.mcda_methods import VIKOR
from crispyn.mcda_methods import VIKOR_SMAA
from crispyn.additions import rank_preferences
from crispyn import correlations as corrs
from crispyn import normalizations as norms
from crispyn import weighting_methods as mcda_weights


# Functions for visualizations

def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 1, 1)
    stacked = True
    width = 0.5
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
    elif x_name == 'Alternative':
        pass
    else:
        df_plot = df_plot.T
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results/bar_chart_weights_' + x_name + '.pdf')
    plt.savefig('results/bar_chart_weights_' + x_name + '.eps')
    plt.show()


def draw_heatmap(data, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods

    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient

    Examples
    ----------
    >>> draw_heatmap(data, title)
    """

    plt.figure(figsize = (6, 4))
    sns.set(font_scale=1.0)
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Weighting methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.savefig('results/heatmap_weights.pdf')
    plt.savefig('results/heatmap_weights.eps')
    plt.show()

def draw_heatmap_smaa(data, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods

    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient

    Examples
    ----------
    >>> draw_heatmap(data, title)
    """

    sns.set(font_scale=1.0)
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlBu_r",
                        linewidth=0.05, linecolor='w')
    plt.yticks(rotation=0)
    plt.ylabel('Alternatives')
    plt.tick_params(labelbottom=False,labeltop=True)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig('results/heatmap_smaa.pdf')
    plt.savefig('results/heatmap_smaa.eps')
    plt.show()


def plot_boxplot(data):
    """
    Display boxplot showing distribution of criteria weights determined with different methods.

    Parameters
    ----------
        data : dataframe
            dataframe with correlation values between compared rankings

    Examples
    ---------
    >>> plot_boxplot(data)
    """
    
    df_melted = pd.melt(data)
    plt.figure(figsize = (7, 4))
    ax = sns.boxplot(x = 'variable', y = 'value', data = df_melted, width = 0.6)
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    ax.set_xlabel('Criterion', fontsize = 12)
    ax.set_ylabel('Different weights distribution', fontsize = 12)
    plt.tight_layout()
    plt.savefig('results/boxplot_weights.pdf')
    plt.savefig('results/boxplot_weights.eps')
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



# main
def main():
    # Load data from CSV
    filename = 'dataset_cars.csv'
    data = pd.read_csv(filename, index_col = 'Ai')
    # Load decision matrix from CSV
    df_data = data.iloc[:len(data) - 1, :]
    # Criteria types are in the last row of CSV
    types = data.iloc[len(data) - 1, :].to_numpy()

    # Convert decision matrix from dataframe to numpy ndarray type for faster calculations.
    matrix = df_data.to_numpy()

    # Symbols for alternatives Ai
    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]
    # Symbols for columns Cj
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, data.shape[1] + 1)]

    
    # Part 1 - study with single weighting method
    
    # Determine criteria weights with chosen weighting method
    weights = mcda_weights.entropy_weighting(matrix)

    # Create the VIKOR method object
    # vikor = VIKOR(normalization_method=norms.minmax_normalization)
    vikor = VIKOR()
    
    # Calculate alternatives preference function values with VIKOR method
    pref = vikor(matrix, weights, types)

    # when there is only one (single) preference vector
    rank = rank_preferences(pref, reverse = False)

    print(rank)

    
    # Part 2 - study with several weighting methods
    # Create a list with weighting methods that you want to explore
    weighting_methods_set = [
        mcda_weights.entropy_weighting,
        #mcda_weights.std_weighting,
        mcda_weights.critic_weighting,
        mcda_weights.gini_weighting,
        mcda_weights.merec_weighting,
        mcda_weights.stat_var_weighting,
        #mcda_weights.cilos_weighting,
        mcda_weights.idocriw_weighting,
        mcda_weights.angle_weighting,
        mcda_weights.coeff_var_weighting
    ]
    

    #df_weights = pd.DataFrame(weights.reshape(1, -1), index = ['Weights'], columns = cols)
    # Create dataframes for weights, preference function values and rankings determined using different weighting methods
    df_weights = pd.DataFrame(index = cols)
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    # Create the VIKOR method object
    vikor = VIKOR()

    for weight_type in weighting_methods_set:
        
        if weight_type.__name__ in ["cilos_weighting", "idocriw_weighting", "angle_weighting", "merec_weighting"]:
            weights = weight_type(matrix, types)
        else:
            weights = weight_type(matrix)

        df_weights[weight_type.__name__[:-10].upper().replace('_', ' ')] = weights
        pref = vikor(matrix, weights, types)

        rank = rank_preferences(pref, reverse = False)

        df_preferences[weight_type.__name__[:-10].upper().replace('_', ' ')] = pref
        df_rankings[weight_type.__name__[:-10].upper().replace('_', ' ')] = rank
        

    df_rankings.to_csv('results/rankings.csv')
    # plot criteria weights distribution using box chart
    plot_boxplot(df_weights.T)

    # plot stacked column chart of criteria weights
    plot_barplot(df_weights, 'Weighting methods', 'Weight value', 'Criteria')

    # plot column chart of alternatives rankings
    plot_barplot(df_rankings, 'Alternatives', 'Rank', 'Weighting methods')

    # Plot heatmaps of rankings correlation coefficient
    # Create dataframe with rankings correlation values
    results = copy.deepcopy(df_rankings)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results[i], results[j]))
            
    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # Plot heatmap with rankings correlation
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')
    

    
    # SMAA method
    cols_ai = [str(el) for el in range(1, matrix.shape[0] + 1)]

    # criteria number
    n = matrix.shape[1]
    # SMAA iterations number (number of weight vectors for SMAA)
    iterations = 10000

    start = time.time()
    # create the VIKOR_SMAA object
    vikor_smaa = VIKOR_SMAA()
    # generate matrix with weight vectors for SMAA
    weight_vectors = vikor_smaa._generate_weights(n, iterations)

    # run the VIKOR_SMAA method
    rank_acceptability_index, central_weight_vector, rank_scores = vikor_smaa(matrix, weight_vectors, types)

    end = time.time() - start
    print('Run time: ', end)
    
    acc_in_df = pd.DataFrame(rank_acceptability_index, index = list_alt_names, columns = cols_ai)
    acc_in_df.to_csv('results_smaa/ai.csv')

    matplotlib.rcdefaults()
    plot_barplot(acc_in_df, 'Alternative', 'Rank acceptability index', 'Rank')

    draw_heatmap_smaa(acc_in_df, 'Rank acceptability indexes')

    central_weights_df = pd.DataFrame(central_weight_vector, index = list_alt_names, columns = cols)
    central_weights_df.to_csv('results_smaa/cw.csv')

    rank_scores_df = pd.DataFrame(rank_scores, index = list_alt_names, columns = ['Rank'])
    rank_scores_df.to_csv('results_smaa/fr.csv')
    


if __name__ == '__main__':
    main()