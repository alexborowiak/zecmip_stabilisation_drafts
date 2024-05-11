# Standard library imports
import sys
import logging
from typing import Dict, List, Union, Optional, Callable, Tuple

# Third-party library imports
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib import ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Local imports (if applicable)
import utils
# logging.basicConfig(format="%(message)s", filemode='w', stream = sys.stdout)
# logger = logging.getLogger()

logger = utils.get_notebook_logger()

sys.path.append('../')
import constants
import plotting_functions
from listXarray import listXarray

from classes import ObsoleteFunctionError


ArrayLike = List[float]
# Usually use a red cmap, so making sure the lines are not red.



REGION_STYLE_DICT = {
    'land': {'color': 'brown', 'linestyle': '--'}, 
    'ocean': {'color': 'blue', 'linestyle': '--'},
    'nh': {'color': 'm', 'linestyle': (0, (1, 1))}, 
    'sh': {'color': 'orange', 'linestyle': (0, (1, 1))},
    'tropics': {'color': 'darkgreen', 'linestyle': '--'}, 
    'mid_nh': {'color': 'red', 'linestyle': '-'}, 
    'mid_sh': {'color': 'tomato', 'linestyle': '--'},
    'arctic': {'color': 'lightblue', 'linestyle': '-'}, 
    'antarctic': {'color': 'darkblue'},
    'global_warm': {'color': 'darkred'},
    'global_cool': {'color': 'darkblue'},
    'gl': {'color': 'black', 'linestyle': '-', 'linewidth': 5, 'zorder': 100}
}


NO_RED_COLORS = ('k', 'green','m', 'mediumpurple', 'black',
                 'lightgreen','lightblue', 'greenyellow')



MODEL_PROFILES = { 'zecmip': constants.ZECMIP_MODEL_PARAMS, 'region': REGION_STYLE_DICT}

# experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
#                      'pr_global': 'brown', 'pr_land_global': 'peru', 
#                     'sic_sea_global': 'blue', 'sic_sea_northern_hemisphere': 'darkblue',
#                        'sic_sea_southern_hemisphere': 'cornflowerblue', 'tos_sea_global': 'orange'}

experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
                     'pr_global': 'green', 'pr_land_global': 'yellowgreen', 
                     'tos_sea_global': 'blue'}



colors = [(1, 1, 1, 0), (0, 0, 0, 1)]  # RGBA format: (red, green, blue, alpha)
black_white_cmap = mcolors.ListedColormap(colors)

def format_plot(fig, ax):
    '''
    Small function for formatting map plots
    Reseson
    ------
    Usef in 07_exploring_consecutive_metrics_all_models_(nb_none)
    '''
    ax.coastlines(alpha=0.7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.left_labels = False
    gl.top_labels = False


# def format_colorbar(pcolor, gs=None, cax:plt.Axes=None, tick_symbol:str='%'):
#     '''
#     Creates a colorbar that takes up all columns in the 0th row.
#     The tick labels are percent
    
#     Reason
#     ------
#     In 07_exploring_consecutive_metrics_all_models_(nb_none) a colorbar 
#     of this type is used repeatedly. 
#     '''
#     cax = plt.subplot(gs[0,:]) if not cax else cax

#     cbar = plt.colorbar(pcolor, cax=cax, orientation='horizontal')
#     xticks = cbar.ax.get_xticks()
#     cbar.ax.set_xticks(xticks)
#     if tick_sybmol: cbar.ax.set_xticklabels([str(int(xt)) + tick_symbol for xt in xticks]);
#     cbar.ax.tick_params(labelsize=labelsize)
    
#     return cbar

def highlight_plot(ax, ds, ds_highlight=None, legend_on:bool =True, yaxis_right:bool=False, label=None,
                  color='tomato', highlight_color='darkred', bbox_to_anchor = [-0.03, 1]):
    '''Plots a line a dash line with the option of another solid line being plotted over the top'''
    if yaxis_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        color = 'green'
        highlight_color = 'darkgreen'
        bbox_to_anchor = [-0.03, 0.8] # 1.19 for rhs
        

    ax.plot(ds.time.values, ds.values,
             color = color, alpha = 0.4, label  = 'Unstable', linestyle='--')
    
    if isinstance(ds_highlight, xr.DataArray):
        ax.plot(ds_highlight.time.values, ds_highlight.values,
                 color = highlight_color, alpha = 0.8, label  = 'Stable')
    else:
        legend_on = False # Turn legend off if only one line
    c1 = plt.gca().lines[0].get_color()
    ax.set_ylabel(label, fontsize = 18,
                   color = c1, rotation = 0, labelpad = 55);
    
    if legend_on:
        leg = ax.legend(ncol=1, fontsize=15, bbox_to_anchor=bbox_to_anchor, frameon=True)
        leg.set_title(label)
        leg.get_title().set_fontsize('15')
        
    major_ticks, minor_ticks = utils.get_tick_locator(ds.values)
    
    
    ax.yaxis.set_major_locator(mticker.MultipleLocator(major_ticks))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(minor_ticks))
    ax.tick_params(axis = 'y', labelsize=14, labelcolor = c1)
    ax.tick_params(axis='x', labelsize=14)
    
    ax.set_xlim(ds.time.values[0], ds.time.values[-1])
    ax.set_xlabel('Time After Emission Cessation (Years)', fontsize=18)


def two_line_highlight_plot(left_ds:xr.DataArray=None, 
                            right_ds:xr.DataArray=None,
                            left_highlight_ds:xr.DataArray=None,
                            right_highlight_ds:xr.DataArray=None, 
                            left_label = None, right_label=None,
                            bounds:Dict[str, float] = None):
    ##plt.style.use('seaborn-darkgrid')

    fig = plt.figure(figsize=  (15,10))
    ax1 = fig.add_subplot(111)
    
    if isinstance(left_ds, xr.DataArray):
        highlight_plot(ax1, left_ds, ds_highlight = left_highlight_ds, label=left_label)
        
    if isinstance(right_ds, xr.DataArray):
        ax2 = ax1.twinx()
        highlight_plot(ax2, right_ds, ds_highlight = right_highlight_ds,
                       yaxis_right=True, label=right_label)
    else:
        ax2=None
        
    if isinstance(bounds, dict):
        for key, value in bounds.items():
            ax1.plot([left_ds.time.values[0], left_ds.time.values[-1]], [value, value], 
                   color='tomato', linestyle=':', alpha=0.8)
        
    return fig, ax1, ax2




def temperature_vs_sn_plot(ax,
                           sn:xr.DataArray=None,
                           temp:xr.DataArray=None,
                           temp_highlight:xr.DataArray=None,
                           sn_highlight:xr.DataArray=None,
                          bounds:Dict[str, float] = None):
    print('!!!!!!! Warning: This is a legacy function and is no longer supported.')
    print('Please use two_line_highlight_plot is sn_plotting')
    ##plt.style.use('seaborn-darkgrid')

    if isinstance(sn, xr.DataArray):
        highlight_plot(ax, sn, ds_highlight = sn_highlight,
                       label='Signal\to\nNoise')
    
    ax2 = ax.twinx()
    highlight_plot(ax2, temp, ds_highlight = temp_highlight,
                   yaxis_right=True, label='GMST\nAnomaly'+ r' ($^{\circ}$C)')
    
    if isinstance(bounds, dict):
        for key, value in bounds.items():
            ax.plot([sn.time.values[0], sn.time.values[-1]], [value, value], 
                   color='tomato', linestyle=':', alpha=0.8)
    
    return ax, ax2



plot_kwargs = dict(height=12, width=22, hspace=0.3, #vmin=-8, vmax=8, step=2, 
                   cmap = 'RdBu_r', line_color = 'limegreen', line_alpha=0.65, 
                   ax2_ylabel = 'Anomaly', cbar_label = 'Signal-to-Noise', cbartick_offset=0,
                   axes_title='', 
                   title='', label_size=12, extend='both', xlowerlim=None, xupperlim=None,  filter_max=True,)


def format_ticks_as_years(ax, xvalues, major_base:int=10, minor_base:int=5, logginglevel='ERROR'):

    utils.change_logginglevel(logginglevel)

    logger.info('Calling function format_ticks_as_years')
    logger.debug(f'{xvalues=}')
    if isinstance(xvalues[0] , cftime.datetime):
        xlabels = xvalues.dt.year.values-1 # The ticks neet to be set back one year so year 1 is year 0
    else:
        xlabels=xvalues

    logger.debug(f'{xlabels=}')
    
    # Set major ticks every 10 units
    logger.info(f'{major_base=}')
    major_locator = mticker.MultipleLocator(base=major_base)
    ax.xaxis.set_major_locator(major_locator)
        
    # Set minor ticks every 5 units
    logger.info(f'{minor_base=}')
    minor_locator = mticker.MultipleLocator(base=minor_base)
    ax.xaxis.set_minor_locator(minor_locator)

def format_xticks(ax, locations, labels, increment:int=10):
    ax.set_xticks(locations[::increment])
    ax.set_xticklabels(labels[::increment])

def plot_all_coord_lines(da: xr.DataArray, coord='model', exp_type=None,
                         fig=None, ax:plt.Axes=None, figsize:tuple=(15,7),
                         font_scale=1, consensus=True, xlabel=None, ylabel=None, yticks_right:list=None, labelpad=60,
                         bbox_to_anchor=(1.02,1), ncol=4, add_legend=True, xlim=None, ylim=None, title=None, 
                         increment:int=10, linestyle='-', colors:Union[str, Tuple, Dict]=None, logginglevel='ERROR',params=None,
                        **kwargs):
    '''
    Plots all of the values in time for a coordinate. E.g. will plot all of the models values
    in time for the global average or for a given grid cell.
    '''

    logger.debug(locals())
    utils.change_logging_level(logginglevel)
    
    fig = plt.figure(figsize=figsize) if not fig else fig
    ax = fig.add_subplot(111) if not ax else ax
    
    coord_values = da[coord].values.flatten() # Flatten in-case 0D array
    
    time = da.time.values

    # CFTIME
    if isinstance(time[0], cftime.datetime): time  = da.time.dt.year.values

    # I ahve created color profiles for all the models 
    if exp_type:
        logger.debug(f'{exp_type=}')
        params = MODEL_PROFILES[exp_type]
        coord_values = [cv for cv in list(params) if cv in coord_values]

    # Consensus needs to go first. So that it appears first in the legend
    if consensus and len(coord_values) > 1:
        ax.plot(time, da.median(dim=coord).values, 
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=500, label='Median', linewidth=4,  
                c='black')
        
    for i, coord_value in enumerate(coord_values):
        logger.debug(f'{i=}, {coord_value=}')

        linewidth = 2
        zorder  = 100
        if exp_type or params:
            logger.info('Using custom params')
            c = params[coord_value]['color']
            ls =  params[coord_value].get('linestyle', '-')
            linewidth = params[coord_value].get('linewidth', 3)
            if 'zorder' in params[coord_value]: zorder = params[coord_value]['zorder']
            logger.debug(f'   - {c=}, {ls=}')
        else:
            c = NO_RED_COLORS[i]
            if isinstance(linestyle, str): ls=linestyle
            elif isinstance(linestyle, dict): ls = linestyle[coord_value]
            else: ls = linestyle[i]

        da_to_plot = da.loc[{coord:coord_value}].values if len(coord_values) > 1 else da.values

        ax.plot(time, da_to_plot,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=zorder, label=coord_value, linewidth=linewidth,  linestyle=ls, 
                c=c)

    ax.grid(True, linestyle='--', color='gray', alpha=0.2)

    
    if isinstance(xlim, tuple): ax.set_xlim(time[xlim[0]], time[xlim[-1]])
    if isinstance(ylim, tuple): ax.set_ylim(ylim)
    if yticks_right is not None: ax.set_yticks(yticks_right)
    if len(coord_values) > 1 and add_legend:
        leg = ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor,
                        fontsize=kwargs.get('legend_fontsize', 12)*font_scale)
        
        leg.set_title(coord.capitalize())
        leg.get_title().set_fontsize(constants.PlotConfig.legend_title_size*font_scale)
        
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, labelpad=labelpad, xlabelpad=15,
                                               # invisible_spines=['top', 'right']
                                   font_scale=font_scale)

    format_ticks_as_years(ax, da.time)
    # Make ticks as just the year vlaues
    # if isinstance(da.time.values[0] , cftime.datetime):
    #     xlabels = da.time.dt.year.values-1 # The ticks neet to be set back one year so year 1 is year 0
    #     # Set major ticks every 10 units
    #     major_locator = mticker.MultipleLocator(base=10)
    #     ax.xaxis.set_major_locator(major_locator)
        
    #     # Set minor ticks every 5 units
    #     minor_locator = mticker.MultipleLocator(base=5)
    #     ax.xaxis.set_minor_locator(minor_locator)
    # else:
    #     xlabels = time
    # format_xticks(ax, time, xlabels, increment)
    return (fig, ax) if 'leg' not in locals() else (fig, ax, leg)



def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)



def format_colorbar(pcolor, gs=None, cax:plt.Axes=None, tick_symbol:str='%'):
    '''
    Creates a colorbar that takes up all columns in the 0th row.
    The tick labels are percent
    
    Reason
    ------
    In 07_exploring_consecutive_metrics_all_models_(nb_none) a colorbar 
    of this type is used repeatedly. 
    '''
    cax = plt.subplot(gs[0,:]) if not cax else cax

    cbar = plt.colorbar(pcolor, cax=cax, orientation='horizontal')
    xticks = cbar.ax.get_xticks()
    cbar.ax.set_xticks(xticks)
    if tick_sybmol: cbar.ax.set_xticklabels([str(int(xt)) + tick_symbol for xt in xticks]);
    cbar.ax.tick_params(labelsize=labelsize)
    
    return cbar


def plot_heatmap(da:xr.DataArray, fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                 figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                 levels:list=None, vmin=None, vmax=None, step=None, yticks:list=None,
                 xlims:tuple=None, font_scale=1, alpha:float=1, 
                 cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                 tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                 title:str=None, axes_title:str=None, labelpad=100, rotation=0,
                 ylabel='Window Length\n(Years)', xlabel='Time After Emission Cessation (Years)', return_all=True,
                 logginglevel='ERROR', **kwargs):
    '''
    Plots a heatmatp of ds. Lots of options for entering different arguements
    '''
    
    utils.change_logginglevel(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')

    
    figsize = figsize if figsize is not None else (plot_kwargs['width'], plot_kwargs['height'])
    fig = fig if fig is not None else plt.figure(figsize=figsize)
    gs = (gs if gs is not None else gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']+hspace))
    
    ax = ax if ax is not None else fig.add_subplot(gs[0])
    
    if xlims is not None: da = da.isel(time=slice(*xlims))
    if not np.issubdtype(da.time.dtype, np.int64): da['time'] = da.time.dt.year.values
    if max_color_lim: da = da.isel(time=slice(None, max_color_lim))
    
    
    if levels is not None: colormap_kwargs = dict(levels=levels)
    elif vmax is not None and step is not None:
        levels = create_levels(vmin=vmin, vmax=vmax, step=step)
        colormap_kwargs = dict(levels=levels)
    else: colormap_kwargs = dict(robust=True)
    logger.info(f'{colormap_kwargs=}')
        
    # ----> Plotting the heatmaps
    cs = da.plot(ax=ax, cmap=cmap, extend=extend, add_colorbar=False, alpha=alpha, levels=levels) # **colormap_kwargs
    
    # ---> Labelling
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel,labelpad=labelpad, font_scale=font_scale, rotation=rotation)
    format_ticks_as_years(ax, da.time)
    fig.suptitle(title, fontsize=constants.PlotConfig.title_size*font_scale, y=0.92)
    ax.set_title(axes_title, fontsize=constants.PlotConfig.title_size*font_scale)
    if xlims is not None: ax.set_xlim(xlims)
    if yticks is not None: ax.set_yticks(yticks)
 
    # ---> Artist
    if patch: ax.add_artist(Rectangle((max_color_lim, 0), xlims[-1]-max_color_lim, 200, color='grey', alpha=0.2, zorder=-1000))
    
    # ---> Colorbar
    if add_colorbar:
        cax = cax if cax is not None else fig.add_subplot(gs[1])
        cbar = plotting_functions.create_colorbar(
            cs, cax=cax, levels=levels, extend=extend, orientation='horizontal',
            font_scale=font_scale, cbar_title=cbar_label, tick_offset=tick_offset,
            cut_ticks=cut_ticks, logginglevel=logginglevel)
        
    if return_all: return fig, gs, ax, cax


def sn_multi_window_in_time(da:xr.DataArray, exp_type:str=None,
                            temp_da:Union[xr.DataArray, xr.Dataset]=None,
                            stable_point_ds:xr.Dataset=None,
                            fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                            figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                            levels:list=None, vmin=None, vmax=None, step=None, 
                            xlims:tuple=(None,None), font_scale=1.5, yticks:list=None, yticks_right:list=None,
                            cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                            tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                            title:str=None, axes_title:str=None,rotation=0, 
                            ylabel='Window Length\n(Years)', xlabel='Time After Emission Cessation (Years)',
                            ax2_ylabel = 'Anomaly', add_legend=True, labelpad_left=100, labelpad_right=50,
                            bbox_to_anchor=(1, 1.3), stable_year_kwargs=dict(),
                            logginglevel='ERROR', return_all=True):
    '''
    
    '''
    # mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('default')
    utils.change_logging_level(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')
    
     
    # ---> Creating plot
    fig = plt.figure(figsize=(plot_kwargs['width'], plot_kwargs['height'])) if fig is None else fig
    gs = gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']) if gs is None else gs
    ax = fig.add_subplot(gs[0]) if ax is None else ax
    
    # ---> Stable Year 
    stable_year_kwargs = {'color':'k', 'linestyle':':', 'linewidth':2.5, **stable_year_kwargs}
    if stable_point_ds: stable_point_ds.time.plot(y='window', ax=ax, **stable_year_kwargs)

    # ---> Plotting colors
    fig, gs, ax, cax = plot_heatmap(da=da, fig=fig, gs=gs, ax=ax, cax=cax,
                 figsize=figsize, cmap=cmap, extend=extend, max_color_lim=max_color_lim,
                 levels=levels, vmin=vmin, vmax=vmax, step=step, 
                 xlims=xlims, font_scale=font_scale, yticks=yticks,
                 cbar_tile=cbar_tile, tick_labels=tick_labels, add_colorbar=add_colorbar, cbar_label=cbar_label,
                 tick_offset=tick_offset, cut_ticks=cut_ticks, patch=patch, hspace=hspace,
                 title=title, axes_title=axes_title, rotatiogn=rotation, 
                 ylabel=ylabel, xlabel=xlabel, labelpad=labelpad_left, logginglevel=logginglevel)

    # ---> Temperature Anomaly
    if isinstance(temp_da, xr.DataArray):
        ax2 = ax.twinx()
        temp_da = temp_da.isel(time=slice(*xlims))
        if not np.issubdtype(temp_da.time.dtype, np.int64): temp_da['time'] = temp_da.time.dt.year.values-1
        plot_all_coord_lines(da=temp_da, ax=ax2, fig=fig, exp_type=exp_type, add_legend=add_legend,
                             font_scale=font_scale, bbox_to_anchor=bbox_to_anchor, yticks_right=yticks_right)
        
        plotting_functions.format_axis(ax2, xlabel=xlabel, ylabel=ax2_ylabel, font_scale=font_scale, labelpad=labelpad_right, rotation=rotation)
        ax2.set_title(None)
        plotting_functions.match_ticks(ax, ax2, 'left') # Note: Has to be left
        
    if return_all:
        try: return (fig, [ax, ax2, cax])
        except NameError: return (fig, [ax, cax])


def plot_all_period_maps(ds, periods, suptitle = 'Percent of Years That are Stable', cmap = 'RdBu', col_increase = 1,
                         row_increase = 2, 
                        y=0.89):
    '''
    Creates a plot of all the different periods 
    '''
    import utils
    
    data_vars = list(ds.data_vars)
    
    # Rows is the number of period, columns is the length of the data vars
    nrows, ncols = (len(periods) * row_increase, len(data_vars) * col_increase)
    
    
    fig = plt.figure(figsize = (8 * ncols, 5 * nrows))
    fig.suptitle(suptitle, y=y, fontsize=15)

    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios = [.2] + [1] * nrows, hspace=0.4, wspace=0)
    
    plot_num = ncols
    for period in periods:
        for dvar in data_vars:
            ax = fig.add_subplot(gs[plot_num], projection=ccrs.PlateCarree())
            da = ds[dvar].sel(period=period)
            pcolor = da.plot(
                ax=ax, vmin=0, vmax=100, cmap=cmap, extend='neither', add_colorbar=False)

            format_plot(fig, ax)

            formatted_period = utils.convert_period_string(period)
            ax.set_title(f'{formatted_period} {dvar.capitalize()}', fontsize=12);
            plot_num += 1
    
    cbar = format_colorbar(pcolor, gs=gs)
    
    return fig, gs, cbar



def plot_year_of_stability(ds: xr.Dataset, varible_to_loop: str, title:str=None):
    '''
    Plots the year of stability for different window lenght. This can be 
    for any variable that is stored as a coordinte. 
    
    ds: xr.Dataset
        
    
    '''
    
    ##plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for variable in ds[varible_to_loop].values:
        color = experiment_colors[variable]
        label = variable.replace('_', ' - ').replace('sea -', '')
        label = label.replace('- global', '')

        da = ds.sel(variable=variable).time.plot(ax=ax,y='window', label=label,
                                                linewidth=1.5, color=color, alpha=0.8)
    if title is None:
        model = str(ds.model.values)
        ECS = f' (ECS={constants.MODEL_PARAMS[model]["ECS"]}K)'
        title = f'{model} Year of Stabilisation {ECS}'
        
    ax.set_title(title, fontsize=25)
    leg = ax.legend(ncol=1, frameon=True, facecolor='white', fontsize=18) # , bbox_to_anchor=[1, 0.857]
    leg.set_title('Variable')
    leg.get_title().set_fontsize('18')
    ax.set_xlim(-1, np.max(ds.time.values))
    ax.set_ylim(np.min(ds.window.values), np.max(ds.window.values))
    ax.set_xlabel('Year of Stabilisation', fontsize=18)
    ax.set_ylabel('Window Length (years)', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    
    return fig, ax

def local_stabilisation_average_year_and_uncertainty_plot(ds, plot_dict, suptitle=None, cmap='Reds'):
    '''
    ds: xr.Dataset
        Coords: window, lat, lat
        Data vars: median_value, uncertainty
    plot_dict:
        dictionary of values for plot
    '''
    windows = ds.window.values

    fig = plt.figure(figsize=(8.3 * len(windows), 12))
    
    gs = gridspec.GridSpec(3, len(windows), height_ratios=[0.2, 1,1])

    axes = []
    plots = []

    if suptitle:
        fig.suptitle(suptitle, fontsize=25)

    y_axis_kwargs = dict(xy=(-0.05, 0.5), ha='center', va='center', xycoords='axes fraction', 
                       rotation=90, size=18)

    for plot_num, window in enumerate(windows):    

        ax = fig.add_subplot(gs[1, plot_num], projection=ccrs.PlateCarree())
        da = ds.sel(window=window).median_value
        plot = da.plot(ax=ax, cmap=cmap, add_colorbar=False, levels=plot_dict[window]['levels'])

        ax.coastlines()
        ax.set_title(f'{window} Year Window', fontsize=18)
        format_plot(fig, ax)

        if not plot_num:
            ax.annotate('Mean', **y_axis_kwargs)

        axes.append(ax)
        plots.append(plot)

    for plot_num, window in enumerate(windows):

        ax = fig.add_subplot(gs[2, plot_num], projection=ccrs.PlateCarree())
        da = ds.sel(window=window).uncertainty
        plot = da.plot(ax=ax, cmap=cmap, add_colorbar=False)
        ax.coastlines()
        format_plot(fig, ax)

        if not plot_num:
            ax.annotate('Uncertainty', **y_axis_kwargs)

        ax.set_title('')
        axes.append(ax)
        plots.append(plot)


    for plot_num, plot in enumerate(plots[:len(windows)]):
        cax = plt.subplot(gs[0, plot_num])
        cbar = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cbar.ax.set_title('Year of Stabilisation', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
    return fig

def __create_plot_dict(ds:xr.DataArray, dim:str, percentile:float=99, vmin=None, num_steps:int=11, debug=False) -> Dict:
    '''
    Creates a dictionary of all of the levels on dimension
    '''
    dims = list(ds.dims)
    dims_bar_dim = [d for d in dims if d != dim]
    robust_max = ds.reduce(np.nanpercentile,q=percentile, dim=dims_bar_dim).values
    if debug: print(f'{robust_max=}')
    dim_values = ds[dim].values
    
    plot_dict = {}
    for dv, vm in zip(dim_values, robust_max):
        step_size_raw = vm/num_steps
        step_size_actual = np.ceil(step_size_raw)
        vm_actual = step_size_actual*num_steps
        vmin = -vm_actual if vmin is None else vmin
        if debug: print(f'{step_size_raw=} {step_size_actual=} {vmin=} {vm_actual=} {step_size_actual=}')
        plot_dict[dv] = {'levels': np.arange(vmin, vm_actual+step_size_actual, step_size_actual)}
    
    
    return plot_dict


def __make_up_coords(ds: xr.Dataset) -> Tuple[str, xr.Dataset]:
    """
    Adds a made-up coordinate to a dataset and returns the modified dataset along with the made-up coordinate name.
    This is used in the `map_plot_all_for_coords` function when there is not column or row to loop through. One is
    made up.

    Args:
        ds (xr.Dataset): The dataset to which the made-up coordinate will be added.

    Returns:
        Tuple[str, xr.Dataset]: A tuple containing the made-up coordinate name and the modified dataset.
    """
    made_up_name = '_'
    ds =  ds.expand_dims(made_up_name).assign_coords(_=(made_up_name, ['_']))
    return '_', ds

def map_plot_all_for_coords(ds:xr.DataArray, variable:int, column_coord:str=None, row_coord:str=None,
                            column_title_tag:str='', y=0.89, row_labels:List[str]=None, vmin=None, 
                            plot_dict:Dict=None, cmap='Reds', extend='max', hspace=0.2, one_colorbar=False):

    '''
    Map plot for two coords
    Updated: 27-th March 2023
    '''
    if isinstance(ds, xr.Dataset): ds = ds.to_array().squeeze()
    
    # Yes, this is weird. Looping over rows and cols later. If not colr
    # or row lets just make one up so this all works.
    if column_coord is None: column_coord, ds = __make_up_coords(ds)
    if row_coord is None: row_coord, ds = __make_up_coords(ds)
        
    
    column_coord_values = ds[column_coord].values; row_coord_values = ds[row_coord].values
    row_labels = row_coord_values if row_labels is None else row_labels
    num_cols = len(column_coord_values); num_rows = len(row_coord_values)
    plot_dict = __create_plot_dict(ds, column_coord, vmin=vmin) if not plot_dict else plot_dict

    fig = plt.figure(figsize=(6*num_cols, 4*num_rows))
    gs = gridspec.GridSpec(num_rows+1, num_cols, height_ratios=[0.2]+[1]*num_rows,
                           hspace=hspace, wspace=0.2)

    fig.suptitle(f'{constants.VARIABLE_INFO[variable]["longname"]} Year of Stabilisation', 
                fontsize=25, y=y)

    axes = []; plots = []

    y_axis_kwargs = dict(xy=(-0.05, 0.5), ha='center', va='center', xycoords='axes fraction', 
                         rotation=90, size=18)
    
    for row, rcv in enumerate(row_coord_values):
        for col, ccv in enumerate(column_coord_values):
            ax = fig.add_subplot(gs[row+1, col], projection=ccrs.PlateCarree())
            da = ds.loc[{column_coord:ccv, row_coord:rcv}]
            plot = da.plot(ax=ax, cmap=cmap, levels = plot_dict[ccv]['levels'], 
                           add_colorbar=False, extend=extend)
            
            if not col: ax.annotate(f'{row_labels[row]}', **y_axis_kwargs)
            format_plot(fig, ax); ax.coastlines(); ax.set_title(None)
            axes.append(ax); plots.append(plot)

    for column_coord,ax in zip(column_coord_values, axes[:len(column_coord_values)]):
        ax.set_title(f'{column_coord} {column_title_tag}', fontsize=18)

    for plot_num, plot in enumerate(plots[:len(column_coord_values)]):            
        cax = plt.subplot(gs[0, plot_num] if not one_colorbar else gs[0, :])
        cbar = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cbar.ax.set_title('Year of Stabilisation', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        if one_colorbar: break
            
    for num, ax_ in enumerate(axes):
        plotting_functions.add_figure_label(ax_, f'{chr(97+num)})')
        
    return fig, axes, plots



def plot_stippled_data(da, ax, stiple_reduction:bool=True, sig_size=2.5, alpha:float=1):
    """
    Plots stippled data on a given axis.

    Args:
        da (xr.DataArray): The data array containing the values to plot.
        ax (plt.Axes): The matplotlib axis object to plot on.
        stiple_reduction (bool): Thin the stippling amount by 1/4.
        sig_size (float): The size of the stipples (default: 2.5).
        alpha (float): The transparency of the stipples (default: 1). Values between 0 and 1

    Returns:
        None
    """

    # The result will be scattered, so we need a meshgrid.
    X,Y = np.meshgrid(da.lon, da.lat)

    #Non-nan values (finite values) are getting replaced by 1. 
    sig = da.where(~np.isfinite(da), 1)

    # All the values that are nan will be replaced with 0. 
    size = np.nan_to_num(sig.values, 0)

    if stiple_reduction:
        size[::2] = 0; size = np.transpose(size)
        size[::2] = 0; size = np.transpose(size)
    ax.scatter(X,Y, s = size*sig_size, color='grey', alpha=alpha, transform=ccrs.PlateCarree())

def map_plot_all_for_coords_2(*args, **kwargs):
    raise ObsoleteFunctionError('This fucntion is obsolete. Please use `map_plot_all_for_coords_2` instead')

    






def map_plot_all_for_coords_3(da: xr.DataArray, levels: ArrayLike, dim: str=None, ncols:int = 3,
                              fig: Optional[plt.Figure] = None, axes: Optional[List[plt.Axes]] = None,
                              gs: Optional[gridspec.GridSpec] = None, cax: Optional[plt.Axes] = None,
                              add_colorbar: bool = True, projection:Callable = ccrs.Robinson,
                              cmap: str = 'RdBu_r', extend: str = 'both', font_scale:float = 1.4,
                              cbar_title: Optional[str] = None, return_all:bool = True, add_label: bool = True,
                              title_tag:str='', add_title=True, ptype='contourf',
                              debug: bool = False, max_stabilisation_year:Optional[int] = None,
                              stabilisation_method: Optional[str] = None, wspace=0.1, 
                              stipling_da: Optional[xr.DataArray] = None,
                              logginglevel='ERROR') -> Optional[Tuple[plt.Figure, List[plt.Axes], plt.Axes]]:
    '''
    Plots rows x columns of the data for the dim coordinate. This can take any projection and
    can also take a new figure or generate a figure.

    Parameters:
        da (xr.DataArray): The data array to plot.
        dim (str): The dimension along which to plot the data.
        levels (ArrayLike): The contour levels to use for plotting.
        ncols (int): Number of columns for the plot grid (default: 3).
        fig (plt.Figure, optional): The matplotlib figure object to use for the plot (default: None).
        axes (List[plt.Axes], optional): List of matplotlib axes objects to use for the subplots (default: None).
        gs (gridspec.GridSpec, optional): The gridspec object defining the layout of subplots (default: None).
        cax (plt.Axes, optional): The axes object to use for the colorbar (default: None).
        add_colorbar (bool): Whether to add a colorbar to the plot (default: True).
        projection (callable): The projection to use for the plot (default: ccrs.Robinson).
        cmap (str): The colormap to use for the plot (default: 'RdBu_r').
        extend (str): The colorbar extension option (default: 'both').
        font_scale (float): The scaling factor for the font sizes (default: 1.4).
        cbar_title (str, optional): The title for the colorbar (default: None).
        return_all (bool): Whether to return all created objects (default: True).
        add_label (bool): Whether to add labels to the subplots (default: True).
        debug (bool): Whether to print debug information (default: False).
        max_stabilisation_year (int, optional): The maximum stabilization year (default: None).
        stabilisation_method (str, optional): The method for stabilization (default: None).
        stipling_da (xr.DataArray, optional): The data array for stippling (default: None).

    Returns:
        If `return_all` is True, the function returns the created figure, axes, and colorbar objects. Otherwise, None.

    '''
    utils.change_logginglevel(logginglevel)
    # Calculate the central longitude based on the input data.
    # If the input data is an xarray DataArray or Dataset, use the mean of the longitude values.
    # Otherwise, if it's an object of type `listXarray`, use the mean of its `lon` values obtained from `single_xarray()`.
    central_longitude = int(np.mean(da.lon.values)) if isinstance(da, (xr.DataArray, xr.Dataset)) else int(np.mean(da.single_xarray().lon.values))

 
    # Print the type of the input data for debugging purposes.
    logger.info(f'Input type = {type(da)}')

    # Check the type of the input data and get the values of the specified dimension (`dim`) accordingly.
    if isinstance(da, (xr.DataArray, xr.Dataset)): 
        if dim is None: raise TypeError(f'When using xarray types dim must not be none ({dim=})')
        dim_vals = da[dim].values
    elif isinstance(da, listXarray): dim_vals = da.refkeys
    else: raise TypeError(f"Expected 'da' to be of type xr.DataArray, xr.Dataset, or listXarray. Got {type(da)} instead.")
    logger.debug(f' - {dim_vals=}')

    # Calculate the number of rows based on the number of dimension values and the specified number of columns (`ncols`).
    nrows = int(np.ceil(len(dim_vals)/ncols))
    logger.debug(f' - {nrows=}')

    # If `fig` is not provided, create a new figure with the specified size (default: 18*nrows x 10*ncols).
    # If `fig` is already provided, use the existing figure.
    fig = plt.figure(figsize=(18*nrows, 10*ncols)) if fig is None else fig

    # If `gs` (GridSpec) is not provided, create a new GridSpec with the specified number of rows and columns,
    # along with height ratios for each row.
    # If `gs` is already provided, use the existing GridSpec.
    gs = gridspec.GridSpec(nrows+1, ncols, height_ratios=[1]*nrows + [0.15], wspace=wspace) if gs is None else gs

    # Calculate the total number of plots.
    num_plots = ncols * nrows

    # Create a list of subplots (axes) based on the number of plots calculated above.
    # If `axes` is already provided, use the existing list of axes.
    axes = ([fig.add_subplot(gs[i], projection=projection(central_longitude=central_longitude)) for i in range(0, num_plots)]
            if axes is None else axes)

    # Initialize a flag to keep track of whether the colorbar has been added already.
    colobar_completed = False

    # Check if the levels for all plots are the same (a single set of levels) or different (individual levels for each plot).
    logger.debug(f' - {levels=}')
    matching_levels_for_all = isinstance(levels[0], (int, float, np.int64, np.float32, np.float64))
    logger.debug(f' - {matching_levels_for_all=}')

    cbars = []
    # Loop through each dimension value and create contour plots for each one.
    logger.info('=>Starting plot loop\n')
    for num, dv in enumerate(dim_vals):
        logger.info(f'{num=} - {dv=}')
        ax = axes[num]
        levels_to_use = levels if matching_levels_for_all else levels[num]
        logger.debug(f' - {levels_to_use=}')

        # Extract the data corresponding to the current dimension value.
        if isinstance(da, (xr.DataArray, xr.Dataset)): da_to_use = da.loc[{dim: dv}]
        elif isinstance(da, listXarray): da_to_use = da[dv]
        
        da_to_use = da_to_use.squeeze()
        logger.debug(da_to_use)

        # Create a filled contour plot on the current subplot (ax).
        plot_kwargs = dict(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend,
                           add_colorbar=False)
        if ptype == 'contourf': c = da_to_use.plot.contourf(**plot_kwargs)
        elif ptype == 'imshow': c = c = da_to_use.plot(**plot_kwargs)
        else: raise TypeError(f'ptype must be one of [contourf, imshow]. Entered {ptype=}')

        
                        # c = ax.contourf(da_to_use.lon.values, da_to_use.lat.values, da_to_use.values,
            #             transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend)


        # Optionally, if `max_stabilisation_year` is provided, overlay a blackout or stipple plot for specific data values.
        # The method of overlay is determined by `stabilisation_method`.
        # Note: The details of the `plot_stippled_data` function are not provided here.
        if max_stabilisation_year:
            da_binary = xr.where(da_to_use > max_stabilisation_year, 1, 0)
            if stabilisation_method == None:
                pass
            elif stabilisation_method == 'blackout':
                ax.contourf(da_binary.lon.values, da_binary.lat.values, da_binary.values,
                            transform=ccrs.PlateCarree(), cmap=black_white_cmap, levels=[0, 0.5, 1], extend='neither')
            elif stabilisation_method == 'stipple':
                plot_stippled_data(da_binary, ax)
            else:
                raise ValueError(f'stabilisation_method must be one of [blackout, stipple]. Value entered {stabilisation_method}')
        utils.change_logginglevel(logginglevel)
        # Optionally, if `stipling_da` is provided, overlay stipple data on the plot.
        # The details of the `plot_stippled_data` function are not provided here.
        if isinstance(stipling_da, xr.DataArray) or isinstance(stipling_da, listXarray):
            if isinstance(stipling_da, xr.DataArray): stippling_data_to_use = stipling_da.loc[{dim: dv}]
            elif isinstance(da, listXarray): stippling_data_to_use = stipling_da[dv]
            plot_stippled_data(stippling_data_to_use, ax, sig_size=.7, stiple_reduction=1, alpha=0.8)

        # Add coastlines to the plot.
        axes[num].coastlines()

        # Set the title for the current subplot with the corresponding dimension value (`dv`).
        if add_title: axes[num].set_title(f'{dv}{title_tag}', fontsize=constants.PlotConfig.title_size*font_scale)

        # Optionally, add a figure label (e.g., "a)", "b)", etc.) to each subplot if `add_label` is True.
        if add_label: plotting_functions.add_figure_label(axes[num], f'{chr(97+num)})', font_scale=font_scale)

        # If `add_colorbar` is True and the colorbar hasn't been added yet (for cases with individual levels),
        # create and add a colorbar to the plot.
        
        if add_colorbar and not colobar_completed:
            gs_to_use = gs[nrows, :] if matching_levels_for_all else gs[nrows, num]  # One cax, or multiple
            if isinstance(cax, plt.Axes):
                cax_to_use = cax
            else:
                cax_to_use = plt.subplot(gs_to_use) if cax is None else cax[num]
            logger.debug(f' - colorbar: {levels_to_use=}')
            cbar = plotting_functions.create_colorbar(c, cax=cax_to_use, levels=levels_to_use, extend=extend, orientation='horizontal',
                            font_scale=font_scale, cbar_title=cbar_title)
            colobar_completed = True if matching_levels_for_all else False
            cbars.append(cbar)

    # If `return_all` is True, return the figure, GridSpec, and axes objects.
    if return_all:
        to_return = (fig, gs, axes, cbars) if 'cbar' in locals() else (fig, gs, axes)
        return to_return


def plot_stable_year_all_models(ds, fig=None, ax=None, linestyle='solid', exp_type=None, add_legend=True, 
                               legend_loc='right', ncol=1, bbox_to_anchor=None, font_scale=1, labelpad=60,
                               xlabel='Year of Stabilisation', xlabelpad=20, xlim=None):
    ''''
    Plotting the year of stabilisation at each window for different models
    '''
    plt.style.use('default')
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)
    if exp_type:
        information_profile = MODEL_PROFILES[exp_type]
        models = [model for model in list(information_profile) if model in ds.model.values]
    else:
        colors = constants.RANDOM_COLOR_LIST
        models = ds.model.values
      
    
    ds.mean(dim='model').time.plot(ax=ax,y='window', label='Mean', color='k', linewidth=3, 
                                    linestyle='solid') 
    
    for num, model in enumerate(models):
        if not exp_type:
            color = colors[num]; label=model
        else:
            color = information_profile[model]['color']
            ECS = information_profile[model]['ECS']
            label = f'{model} ({ECS=}K)'
        da = ds.sel(model=model).time.plot(ax=ax,y='window', linewidth=2.5, alpha=0.8,
                                           color=color, label=label, linestyle=linestyle)
        
    ylims = np.take(ds.window.values, [0,-1])
    # xlims = [np.min(ds.time.values)-5, np.max(ds.time.values)]
    # ax.set_xlim(xlims)
    if xlim: ax.set_xlim(xlim)
    ax.set_ylim(ylims)
    if isinstance(ncol, str):
        if ncol == 'coords': ncol = len(models)
    if add_legend:
        if isinstance(bbox_to_anchor, tuple): bbox_to_anchor=bbox_to_anchor
        else:
            if legend_loc == 'right': bbox_to_anchor=(1, 0.857)
            elif legend_loc == 'top_ofset': bbox_to_anchor=(1.5, 1.1)
        leg = ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, frameon=True, facecolor='white',
                        fontsize=constants.PlotConfig.legend_text_size*font_scale)
        leg.set_title('Model')
        leg.get_title().set_fontsize(constants.PlotConfig.legend_title_size*font_scale)
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel='Window Length\n(Years)',
                                   font_scale=font_scale, xlabelpad=xlabelpad, labelpad=labelpad,
                                  invisible_spines=['top', 'right'])
    ax.set_title('')
    
    return fig, ax

def plot_average_stable_year(ds1, ds2, fig=None, ax=None, font_scale:float=1,
                             ylabel='Window Length (years)', xlabel='Time (Years)'):
    ''''
    Plotting the median year of stabilisation at each window for two 
    different datasets.
    '''
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)

    ds1_mean = ds1.mean(dim='model')
    ds2_mean = ds2.mean(dim='model')
    ds1_mean.time.plot(ax=ax, y='window', label='rolling', color='k', linewidth=1.5, linestyle='solid') 
    ds2_mean.time.plot(ax=ax,y='window', label='static', color='k', linewidth=1.5, linestyle='dashed') 

    ylims = np.take(ds1_mean.window.values, [0,-1])
    all_x_values = np.concatenate([ds1_mean.time.values, ds2_mean.time.values]).flatten()
    xlims = [np.min(all_x_values), np.max(all_x_values)]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    leg = ax.legend(ncol=1, loc='best', frameon=True, facecolor='white', fontsize=14) 
    leg.set_title('Noise Type')
    leg.get_title().set_fontsize('16')
    
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel,font_scale=font_scale, invisible_spines=['top', 'right'])
    ax.set_title('')
    
    return fig, ax


# def map_plot_all_for_coords_2(da: xr.DataArray, dim: str, levels: ArrayLike, ncols:int = 3,
#                               fig: Optional[plt.Figure] = None, axes: Optional[List[plt.Axes]] = None,
#                               gs: Optional[gridspec.GridSpec] = None, cax: Optional[plt.Axes] = None,
#                               add_colorbar: bool = True, projection:Callable = ccrs.Robinson,
#                               cmap: str = 'RdBu_r', extend: str = 'both', font_scale:float = 1.4,
#                               cbar_title: Optional[str] = None, return_all:bool = True, add_label: bool = True,
#                               debug: bool = False, max_stabilisation_year:Optional[int] = None,
#                               stabilisation_method: Optional[str] = None,
#                               stipling_da: Optional[xr.DataArray] = None) -> Optional[Tuple[plt.Figure, List[plt.Axes], plt.Axes]]:
#     '''
#     Plots rows x columns of the data for the dim coordinate. This can take any projection and
#     can also take a new figure or generate a figure.

#     Parameters:
#         da (xr.DataArray): The data array to plot.
#         dim (str): The dimension along which to plot the data.
#         levels (ArrayLike): The contour levels to use for plotting.
#         ncols (int): Number of columns for the plot grid (default: 3).
#         fig (plt.Figure, optional): The matplotlib figure object to use for the plot (default: None).
#         axes (List[plt.Axes], optional): List of matplotlib axes objects to use for the subplots (default: None).
#         gs (gridspec.GridSpec, optional): The gridspec object defining the layout of subplots (default: None).
#         cax (plt.Axes, optional): The axes object to use for the colorbar (default: None).
#         add_colorbar (bool): Whether to add a colorbar to the plot (default: True).
#         projection (callable): The projection to use for the plot (default: ccrs.Robinson).
#         cmap (str): The colormap to use for the plot (default: 'RdBu_r').
#         extend (str): The colorbar extension option (default: 'both').
#         font_scale (float): The scaling factor for the font sizes (default: 1.4).
#         cbar_title (str, optional): The title for the colorbar (default: None).
#         return_all (bool): Whether to return all created objects (default: True).
#         add_label (bool): Whether to add labels to the subplots (default: True).
#         debug (bool): Whether to print debug information (default: False).
#         max_stabilisation_year (int, optional): The maximum stabilization year (default: None).
#         stabilisation_method (str, optional): The method for stabilization (default: None).
#         stipling_da (xr.DataArray, optional): The data array for stippling (default: None).

#     Returns:
#         If `return_all` is True, the function returns the created figure, axes, and colorbar objects. Otherwise, None.

#     '''
    
#     projection = projection(central_longitude=int(np.mean(da.lon.values)))
    
#     dim_vals = da[dim].values
#     nrows = int(np.ceil(len(dim_vals)/ncols))
#     fig = plt.figure(figsize=(18*nrows, 10*ncols)) if fig is None else fig
#     gs = gridspec.GridSpec(nrows+1, ncols, height_ratios=[1]*nrows + [0.15]) if gs is None else gs
#     num_plots = ncols*nrows#; num_plots = num_plots-1 if num_plots>2 else num_plots
#     axes = ([fig.add_subplot(gs[i], projection=projection) for i in range(0,num_plots)]
#             if axes is None else axes)
#     colobar_completed = False
#     # If levels[0] is not list or array it will be a int/float, then this is just one levels.
#     matching_levels_for_all = isinstance(levels[0], (int, float, np.int64, np.float32, np.float64))
#     for num,dv in enumerate(dim_vals):
#         ax = axes[num]
#         levels_to_use = levels if matching_levels_for_all else levels[num]
#         if debug: print(levels_to_use)
#         da_to_use = da.loc[{dim:dv}]
#         c = ax.contourf(da.lon.values, da.lat.values, da_to_use.values,
#                       transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend)
#         if max_stabilisation_year:
#             da_binary = xr.where(da_to_use > max_stabilisation_year, 1, 0)
#             if stabilisation_method == 'blackout':
#                 ax.contourf(da_binary.lon.values, da_binary.lat.values, da_binary.values,
#                           transform=ccrs.PlateCarree(), cmap=black_white_cmap, levels=[0, 0.5,1], extend='neither')
#             elif stabilisation_method == 'stipple':
#                 plot_stippled_data(da_binary, ax)
#             else: raise ValueError(f'stabilisation_method must be one of [blackout, stipple]. Value entered: {stabilisation_method}')
#         if isinstance(stipling_da, xr.DataArray):
#             stippling_data_to_use = stipling_da.loc[{dim:dv}]
#             plot_stippled_data(stippling_data_to_use, ax, sig_size=.7, stiple_reduction=1, alpha=0.8)
#         axes[num].coastlines()
#         axes[num].set_title(dv, fontsize=constants.PlotConfig.title_size*font_scale)
#         if add_label: plotting_functions.add_figure_label(axes[num], f'{chr(97+num)})', font_scale=font_scale)
#         if add_colorbar and not colobar_completed:
#             gs_to_use = gs[nrows, :] if matching_levels_for_all else gs[nrows, num] # One cax, or multiple
#             print(gs_to_use)
#             if isinstance(cax, plt.Axes):
#                 cax_to_use = cax
#             else: cax_to_use = plt.subplot(gs_to_use) if cax is None else cax[num]
#             create_colorbar(c, cax=cax_to_use, levels=levels_to_use, extend=extend, orientation='horizontal',
#                             font_scale=font_scale, cbar_title=cbar_title)
#             colobar_completed = True if matching_levels_for_all else False

#     if return_all: return fig, gs, axes



#     central_longitude = int(np.mean(da.lon.values)) if isinstance(da, (xr.DataArray, xr.Dataset)) else int(np.mean(da.single_xarray().lon.values))
#     projection = projection(central_longitude=central_longitude)
    
    
#     print(type(da))
#     if isinstance(da, (xr.DataArray, xr.Dataset)):
#         dim_vals = da[dim].values
#     elif isinstance(da, listXarray):
#         dim_vals = da.refkeys
    
        
#     nrows = int(np.ceil(len(dim_vals)/ncols))
#     fig = plt.figure(figsize=(18*nrows, 10*ncols)) if fig is None else fig
#     gs = gridspec.GridSpec(nrows+1, ncols, height_ratios=[1]*nrows + [0.15]) if gs is None else gs
#     num_plots = ncols*nrows#; num_plots = num_plots-1 if num_plots>2 else num_plots
#     axes = ([fig.add_subplot(gs[i], projection=projection) for i in range(0,num_plots)]
#             if axes is None else axes)
#     colobar_completed = False
#     # If levels[0] is not list or array it will be a int/float, then this is just one levels.
#     matching_levels_for_all = isinstance(levels[0], (int, float, np.int64, np.float32, np.float64))
#     for num,dv in enumerate(dim_vals):
#         ax = axes[num]
#         levels_to_use = levels if matching_levels_for_all else levels[num]
#         if debug: print(levels_to_use)
#         if isinstance(da, (xr.DataArray, xr.Dataset)):
#             da_to_use = da.loc[{dim:dv}]
#         elif isinstance(da, listXarray):
#             da_to_use = da[dv]
        
#         c = ax.contourf(da_to_use.lon.values, da_to_use.lat.values, da_to_use.values,
#                       transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend)
#         if max_stabilisation_year:
#             da_binary = xr.where(da_to_use > max_stabilisation_year, 1, 0)
#             if stabilisation_method == 'blackout':
#                 ax.contourf(da_binary.lon.values, da_binary.lat.values, da_binary.values,
#                           transform=ccrs.PlateCarree(), cmap=black_white_cmap, levels=[0, 0.5,1], extend='neither')
#             elif stabilisation_method == 'stipple':
#                 plot_stippled_data(da_binary, ax)
#             else: raise ValueError(f'stabilisation_method must be one of [blackout, stipple]. Value entered: {stabilisation_method}')
#         if isinstance(stipling_da, xr.DataArray) or isinstance(stipling_da, listXarray):
#             if isinstance(stipling_da, xr.DataArray):
#                 stippling_data_to_use = stipling_da.loc[{dim:dv}]
#             elif isinstance(da, listXarray):
#                 stippling_data_to_use = stipling_da[dv]            
#             plot_stippled_data(stippling_data_to_use, ax, sig_size=.7, stiple_reduction=1, alpha=0.8)
#         axes[num].coastlines()
#         axes[num].set_title(dv, fontsize=constants.PlotConfig.title_size*font_scale)
#         if add_label: plotting_functions.add_figure_label(axes[num], f'{chr(97+num)})', font_scale=font_scale)
#         if add_colorbar and not colobar_completed:
#             gs_to_use = gs[nrows, :] if matching_levels_for_all else gs[nrows, num] # One cax, or multiple
#             print(gs_to_use)
#             if isinstance(cax, plt.Axes):
#                 cax_to_use = cax
#             else: cax_to_use = plt.subplot(gs_to_use) if cax is None else cax[num]
#             create_colorbar(c, cax=cax_to_use, levels=levels_to_use, extend=extend, orientation='horizontal',
#                             font_scale=font_scale, cbar_title=cbar_title)
#             colobar_completed = True if matching_levels_for_all else False

#     if return_all: return fig, gs, axes
