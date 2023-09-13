import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from crispyn.mcda_methods import VIKOR
from crispyn.additions import rank_preferences
from crispyn import correlations as corrs
from crispyn import weighting_methods as mcda_weights


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
    ax.set_ylabel('Criteria weights distribution', fontsize = 12)
    plt.title('Distribution of criteria weights values', fontsize = 12)
    plt.tight_layout()
    plt.savefig('results_update/boxplot_weights.pdf')
    plt.savefig('results_update/boxplot_weights.eps')
    plt.show()


# bar (column) chart
def plot_barplot_stacked(df_plot, stacked = False):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.

        stacked : Boolean
            Variable denoting if the chart is to be stacked or not.

    Examples
    ----------
    >>> plot_barplot(df_plot)
    """

    ax = df_plot.plot(kind='bar', width = 0.6, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    if stacked == False:
        list_rank = np.arange(0, 0.30, 0.05)
        ax.set_yticks(list_rank)
        ax.set_ylim(0, 0.25)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    
    ncol = df_plot.shape[1]
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Criteria', fontsize = 12)

    ax.set_xlabel('Weighting methods', fontsize = 12)
    ax.set_ylabel('Weight value', fontsize = 12)

    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results_update/boxplot_weights_col.pdf')
    plt.savefig('results_update/boxplot_weights_col.eps')
    plt.show()


# plot radar chart
def plot_radar(data):
    """
    Visualization method to display rankings of alternatives obtained with different methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing rankings of alternatives obtained with different 
            methods. The particular rankings are contained in subsequent columns of DataFrame.
        
    Examples
    ----------
    >>> plot_radar(data)
    """

    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, '-o', linewidth=2)
        # ax.fill(angles, stats, label='_nolegend_', alpha=0.5)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_rgrids(np.arange(1, data.shape[0] + 1, 1))
    ax.grid(True)
    ax.set_axisbelow(True)
    # plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    if data.shape[1] % 2 == 0:
        ncol = data.shape[1] // 2
    else:
        ncol = data.shape[1] // 2 + 1
    # plt.legend(data.columns, bbox_to_anchor=(-0.1, 1.1, 1.2, .102), loc='lower left',
    # ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weighting methods', fontsize = 12)
    plt.legend(data.columns, bbox_to_anchor=(-0.1, 1.1, 1.2, .102), loc='lower left',
    ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weighting methods', fontsize = 12)
    # plt.title('VIKOR Rankings with different weighting methods')
    # ax.set_title('VIKOR Rankings with different weighting methods', y=1.39)
    plt.tight_layout()
    plt.savefig('results_update/radar.pdf')
    plt.savefig('results_update/radar.eps')
    plt.show()


# plot radar chart
def plot_radar_weights(data):
    """
    Visualization method to display weights values obtained with different weighing methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing weights obtained with different weighting
            methods. The particular weights are contained in subsequent columns of DataFrame.
        
    Examples
    ----------
    >>> plot_radar_weights(data)
    """

    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, linewidth=2)
        # ax.fill(angles, stats, label='_nolegend_', alpha=0.5)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_rgrids(np.round(np.linspace(0, np.max(stats) + 0.05, 5), 2))
    ax.grid(True)
    ax.set_axisbelow(True)
    # plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    if data.shape[1] % 2 == 0:
        ncol = data.shape[1] // 2
    else:
        ncol = data.shape[1] // 2 + 1
    plt.legend(data.columns, bbox_to_anchor=(-0.1, 1.1, 1.2, .102), loc='lower left',
    ncol = ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weighting methods', fontsize = 12)
    plt.tight_layout()
    plt.savefig('results_update/radar_weights.pdf')
    plt.savefig('results_update/radar_weights.eps')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap)
    """
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="PiYG",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.yticks(va="center")
    plt.xlabel('Weighting methods')
    plt.ylabel('Weighting methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.savefig('results_update/heatmap.pdf')
    plt.savefig('results_update/heatmap.eps')
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

    # load data of localisations
    data = pd.read_csv('./dataset_localisations.csv', index_col='Symbol')

    df_data = data.iloc[:len(data) - 1, :]
    types = data.iloc[len(data) - 1, :].to_numpy()
    matrix = df_data.to_numpy()

    # dataframe for weights
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, matrix.shape[1] + 1)]
    df_weights = pd.DataFrame(index = cols)

    # AHP weighting
    # matrix with criteria pairwise comparison
    PCcriteria_ahp = np.array([
        [1, 2, 2, 2, 6, 2, 1, 1, 1/2, 1/2],
        [1/2, 1, 1, 1, 4, 1, 1, 1, 1/2, 1/2],
        [1/2, 1, 1, 1, 3, 1, 1/2, 1/2, 1/3, 1/3],
        [1/2, 1, 1, 1, 3, 1, 1/2, 1/2, 1/3, 1/3],
        [1/6, 1/4, 1/3, 1/3, 1, 1/3, 1/5, 1/5, 1/9, 1/9],
        [1/2, 1, 1, 1, 3, 1, 1/2, 1/2, 1/3, 1/3],
        [1, 1, 2, 2, 5, 2, 1, 1, 1/2, 1/2],
        [1, 1, 2, 2, 5, 2, 1, 1, 1/2, 1/2],
        [2, 2, 3, 3, 9, 3, 2, 2, 1, 1],
        [2, 2, 3, 3, 9, 3, 2, 2, 1, 1]
    ])


    crit_ahp = pd.DataFrame(PCcriteria_ahp)
    crit_ahp.to_csv('./results_update/crit_ahp.csv')

    ahp_weighting = mcda_weights.AHP_WEIGHTING()
    weights_ahp = ahp_weighting(X = PCcriteria_ahp, compute_priority_vector_method=ahp_weighting._eigenvector)
    df_weights['AHP'] = weights_ahp

    # SWARA
    criteria_indexes = np.array([8, 9, 0, 6, 7, 1, 2, 3, 5, 4])
    s = np.array([0, 0.4, 0.17, 0, 0.2, 0.25, 0, 0, 0.67])

    weights_swara = mcda_weights.swara_weighting(criteria_indexes, s)

    df_weights['SWARA'] = weights_swara

    # weights LBWA
    # criteria_indexes, criteria_values_I
    criteria_indexes = [
        [8, 9, 0],
        [6, 7, 1],
        [2, 3, 5],
        [],
        [],
        [],
        [],
        [],
        [],
        [4]
    ]

    criteria_values_I = [
        [0, 0, 2],
        [1, 1, 2],
        [1, 1, 1],
        [],
        [],
        [],
        [],
        [],
        [],
        [3]
    ]

    weights_lbwa = mcda_weights.lbwa_weighting(criteria_indexes, criteria_values_I)

    df_weights['LBWA'] = weights_lbwa

    # SAPEVO
    PCcriteria_sapevo = np.array([
        [ 0, 1, 1, 1, 2, 1, 0, 0, -1, -1],
        [-1, 0, 0, 0, 1, 0, 0, 0, -1, -1],
        [-1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
        [-1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
        [-2, -1, -1, -1, 0, -1, -2, -2, -3, -3],
        [-1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
        [ 0, 0, 1, 1, 2, 1, 0, 0, -1, -1],
        [ 0, 0, 1, 1, 2, 1, 0, 0, -1, -1],
        [ 1, 1, 1, 1, 3, 1, 1, 1, 0, 0],
        [ 1, 1, 1, 1, 3, 1, 1, 1, 0, 0]
    ])

    crit_sapevo = pd.DataFrame(PCcriteria_sapevo)
    crit_sapevo.to_csv('./results_update/crit_sapevo.csv')

    weights_sapevo = mcda_weights.sapevo_weighting(PCcriteria_sapevo)

    df_weights['SAPEVO'] = weights_sapevo

    df_weights.to_csv('./results_update/df_weights.csv')

    plot_boxplot(df_weights.T)
    plot_barplot_stacked(df_weights.T, stacked = True)
    plot_barplot_stacked(df_weights.T, stacked = False)
    plot_radar_weights(df_weights)

    weighting_methods_names = ['AHP', 'SWARA', 'LBWA', 'SAPEVO']
    weights_list = [weights_ahp, weights_swara, weights_lbwa, weights_sapevo]

    # MCDA assessment
    # dataframe for alternatives
    alts = [r'$A_{' + str(j) + '}$' for j in range(1, matrix.shape[0] + 1)]
    df_prefs = pd.DataFrame(index = alts)
    df_ranks = pd.DataFrame(index = alts)

    vikor = VIKOR()
    for el, weights in enumerate(weights_list):
        pref = vikor(matrix, weights, types)
        rank = rank_preferences(pref, reverse=False)

        df_prefs[weighting_methods_names[el]] = pref
        df_ranks[weighting_methods_names[el]] = rank

    plot_radar(df_ranks)

    df_prefs.to_csv('./results_update/df_prefs.csv')
    df_ranks.to_csv('./results_update/df_ranks.csv')
    

    # Rankings correlations
    results = copy.deepcopy(df_ranks)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results[i], results[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')



if __name__ == '__main__':
    main()