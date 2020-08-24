import seaborn as sns
import synapseclient as sc
import pandas as pd
import numpy as np
import pylab as plt
import matplotlib
import matplotlib.gridspec as gridspec
import warnings
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests

class SeabornFig2Grid():
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def get_agg_fog_features(df, freeze_threshold):
    """
    per paper, freeze index more than 2.5 is considered as freezing
    """
    freeze = df.copy()
    freeze["is_freeze_x"] = np.where(freeze["x_energy_freeze_index"] >= freeze_threshold, 1, 0)
    freeze["is_freeze_y"] = np.where(freeze["y_energy_freeze_index"] >= freeze_threshold, 1, 0)
    freeze["is_freeze_z"] = np.where(freeze["z_energy_freeze_index"] >= freeze_threshold, 1, 0)
    freeze["is_freeze_AA"] = np.where(freeze["AA_energy_freeze_index"] >= freeze_threshold, 1, 0)
    
    group = freeze.groupby(["healthCode","recordId"]).agg(
    {"is_freeze_x":"sum", 
     "is_freeze_y":"sum",
     "is_freeze_z":"sum",
     "is_freeze_AA":"sum",
     "window_end":"last",
     "window_start":"first"})
    group["activity_duration"] = group["window_end"] - group["window_start"]
    
    group["freeze_of_gait_per_secs_x"] = group["is_freeze_x"]/group["activity_duration"]
    group["freeze_of_gait_per_secs_y"] = group["is_freeze_y"]/group["activity_duration"]
    group["freeze_of_gait_per_secs_z"] = group["is_freeze_z"]/group["activity_duration"]
    group["freeze_of_gait_per_secs_AA"] = group["is_freeze_AA"]/group["activity_duration"]
    
    freeze_med_feat = group.reset_index()[["healthCode", 
                                           "freeze_of_gait_per_secs_x", 
                                           "freeze_of_gait_per_secs_y", 
                                           "freeze_of_gait_per_secs_z", 
                                           "freeze_of_gait_per_secs_AA"]].groupby("healthCode").median().add_suffix('_agg_med')

    freeze_iqr_feat = group.reset_index()[["healthCode", 
                                           "freeze_of_gait_per_secs_x", 
                                           "freeze_of_gait_per_secs_y", 
                                           "freeze_of_gait_per_secs_z", 
                                           "freeze_of_gait_per_secs_AA"]].groupby("healthCode").agg(iqr).add_suffix('_agg_iqr')
    return freeze_iqr_feat.join(freeze_med_feat)

def get_agg_features(data):
    ## get aggregated data for each record
    iqr_agg_feat = data.groupby(["healthCode"])\
                            .agg(iqr).add_suffix('_agg_iqr')
    med_agg_feat = data.groupby(["healthCode"])\
                            .median().add_suffix('_agg_med')
    counts = data.groupby("healthCode")\
        .agg('nunique')[["createdOn"]]\
        .rename({"createdOn":"nrecords"}, axis = 1)
    return (iqr_agg_feat.join(med_agg_feat)).join(counts)

def iqr(x):
    """
    Function for getting IQR value
    """
    return x.quantile(0.75) - x.quantile(0.25)

def get_pval_and_corr(data, metric):
    p_val = {}
    for feat in feat_used:  
        try:
            res = data[[feat, metric]].dropna()
            scale = (res[feat]- res[feat].min())/(res[feat].max() - res[feat].min())
            corr_test = (stats.pearsonr(scale,  res[metric]))
            p_val[feat] = [corr_test[0], corr_test[1]]
        except:
            pass
    p_val = pd.DataFrame(p_val).T.rename({0:"correlation", 1:"p-value"}, axis = 1)
    adj_pval = multipletests(p_val["p-value"], alpha = 0.05, method = "bonferroni")[1]
    p_val["adjusted-p-value"] = adj_pval
    return(p_val)

def get_p_val_metrics(data, feat_used):
    p_val = {}
    for feat in feat_used:
        control = data[data["PD"] == 0].dropna()[feat]
        PD = data[data["PD"] == 1].dropna()[feat]
        scaled_control = (control - control.min())/(control.max()-control.min())
        scaled_PD = (PD - PD.min())/(PD.max()-PD.min())

        t, pvalue_non_parametric = stats.mannwhitneyu(
            scaled_PD,
            scaled_control, alternative=None)

        t, pvalue_parametric = stats.ttest_ind(
            scaled_PD,
            scaled_control, 
            equal_var = False)
        p_val[feat] = [pvalue_non_parametric, pvalue_parametric]

    p_val = pd.DataFrame(p_val).T.rename({0:"Mann-Whitney", 1:"T-test"}, axis = 1)
    p_val["corrected_p_val"] = multipletests(p_val["T-test"], method = "bonferroni")[1]
    p_val = p_val.sort_values("corrected_p_val")
    return p_val