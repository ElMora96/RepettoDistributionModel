# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:32:15 2021

@author: giamm
"""
#############################################################################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#############################################################################
''' A module that contains various functions for plotting'''

# Default plot parameters 
def_params = {
'figsize': (420/25.4 , 297/25.4),
'orientation': 'horizontal',
'fonts_sml': [18, 20, 22],
}

#############################################################################
def shares_pie_plot(data, 
					values_col = 'share',
					first_group_col = 'player',
					second_group_col = 'type',
					values_threshold = 0.15,
					second_pie_flag=True,
					fig_specs={
						'suptitle': "",},
					**plot_params):
	'''
	The function returns a figure object containing a pie plot showhing the
	distribution of the benefits among different players.
	
	Parameters
	----------
	data: pd.DataFrame
		Contains for each player ('player') the share of benefit ('share') 
		and the type of player ('type').
		(producer, consumer, prosumer,...).
	values_col: str, optional
		Name of the column in data where the shares of the players are stored.
		The default is 'share'.
	first_group_col: str, optional
		Name of the column in data where the names of the players are stored.
		The default is 'player'.
	second_group_col: str, optional
		Name of the column in data where the types of the players are stored.
		The default is 'type'.
	values_threshold: float, optional
		Quantile for grouping data in 'others' depending on their value.
		The default is 0.15.
	second_pie_flag: bool, optional
		 Flag to activate the plot of a second pie where the shares are 
		 divided between producers, prosumers, and consumers.
		 The default is True.
	fig_specs: dict, optional
		Contains a series of specifications for the plot.
		The default is {'suptitle': "",}.
	**plot_params: dict
		Contins various parameters about the plot\'s geometry and font sizes
	
	Returns
	-------
	fig.
	'''
	
	## Plot parameters
	
	# The parameters that are not specified are set to their default values
	for param in def_params: 
		if param not in plot_params: plot_params[param] = def_params[param]
	
	# The values of figsize and orientation are stored in variables
	figsize = plot_params	['figsize']
	orientation = plot_params	['orientation']
	if orientation != 'horizontal': figsize = figsize[::-1]
	
	# The fonts are stored in variables and set for different types of text
	number_of_fonts = len(plot_params['fonts_sml'])
	if number_of_fonts < 1:
		font_small, font_medium, font_large = plot_params['fonts_sml'][0]
	elif number_of_fonts < 2:
		font_small, font_medium = plot_params['fonts_sml'][0]
		font_large = plot_params['fonts_sml'][-1]
	else:
		font_small = plot_params['fonts_sml'][0]
		font_medium = plot_params['fonts_sml'][1]
		font_large = plot_params['fonts_sml'][-1]
	
	fontsize_title = font_large
	fontsize_legend = font_medium
	fontsize_pielabels = font_small
	
	## Data handling
	
	# Raise errore if columns names in dataframe are not consistent
	if (values_col not in data) or (first_group_col not in data) \
		or (second_pie_flag==True and second_group_col not in data):
			error_message = \
				'Columns names in data are not consistant, check funct doc.'
			raise Exception(error_message)
			
	# Normalise data
	values_tot = data[values_col].sum()
	if values_tot!= 1.:
		data.loc[:, values_col] = data[values_col]/values_tot
	
	# In order to avoid too many 'little slices' (i.e., players with small
	# share) in the pie plot, all players whose share is below a certain 
	# threshold are grouped in the 'others' category'
	if not 0 < values_threshold < 1: values_threshold = 0.3
	threshold = data[values_col].quantile(values_threshold)
	
	# If there are more than one player whose share is below the threshold 
	# value, they are grouped in the 'others' category.
	others_flag=False
	if len(data[data[values_col] < threshold]) > 1:
		others_flag=True
		data_others = data[data[values_col] < threshold]
		data_others.loc['total'] = data_others.sum(numeric_only=True, axis=0)
		data_others.fillna(value='others', inplace=True)
		data = data.loc[data[values_col] >= threshold]
		data = data.append(data_others.loc['total'], ignore_index=True)
	
	# The data in the dataframe are sorted. If the flag for the second pie,
	# where the data are plotted for different types of players, is on, the
	# data are sorted by type first, and shares second. Otherwise they are 
	# sorted by the value of the shares.
	if second_pie_flag is True:
		# Storing the second_group column entries as categories to sort the
		# data. The last row ('others') is first removed, in order to sort
		# the entries in alphabetical order and then appended again.
		if others_flag:
			second_group_labels = list(data[second_group_col].unique())[:-1]
			second_group_labels.sort(reverse=True)
			second_group_labels.append('others')
		else:
			second_group_labels = list(data[second_group_col].unique())
		data.loc[:, second_group_col] = pd.Categorical(data[second_group_col],
				   categories=second_group_labels)
		data.sort_values(by=[second_group_col, values_col],
				   ascending=[True, False],
				   inplace=True)
	else:
		data.sort_values(by=values_col, axis=0, ascending=False,
				   inplace=True)
	
	# Number of slices (rows in the dataframe)
	n_slices = len(data)
	
	## Figure and plot
	
	# A new figure is created with the specified size
	fig, ax = plt.subplots(figsize=figsize)
	
	# Suptitle of the figure
	if 'suptitle' in fig_specs: suptitle = fig_specs['suptitle']
	else: suptitle = ""
	fig.suptitle(suptitle, fontsize=fontsize_title, fontweight='bold')
	
	# The geometry is adjusted
	if suptitle == '': top = 0.95
	else: top = 0.9
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=top,
					 wspace=None, hspace=None)
	
	# Colormap to be used for the slices of the pie
	cmap_name = 'tab10' #original was autumns
	cmap = plt.get_cmap(cmap_name)
	colors = [cmap(i) for i in np.linspace(0, 1, n_slices)]
	
	# Data in data are plotted in a pie chart
	labels = list(data[first_group_col])
	labels = [label.title() for label in labels]
	w, l, p = \
		ax.pie(data[values_col], labels=labels,
		 startangle=0, colors=colors, 
		 autopct='%1.1f%%', pctdistance=0.8, radius=0.8, labeldistance=None,
		 frame=False, shadow=False,
		 textprops = {'fontsize':fontsize_pielabels},
		 wedgeprops={'edgecolor':'k', 'linewidth': 1,'linestyle': 'solid',})
	
	# Each pct text is rotated in order to be radially printed, with variable
	# distance from the centre
	pctdists = np.linspace(1, 1, n_slices)
	for t, d in zip(p, pctdists):
		# Position of the text and distance from the centre
		xi,yi = t.get_position()
		ri = np.sqrt(xi**2+yi**2)
		# Angle (rad)
		phi = np.arctan2(yi,xi)
		# New distance from the centre, selected from pctdists
		x = d*ri*np.cos(phi)
		y = d*ri*np.sin(phi)
		t.set_position((x,y))
		# Rotation of the text, radially rotated (°)
		rotation = phi/np.pi * 180
		tol = 1e-1
		if (rotation-90 >= -tol) or (rotation+90 <= tol):
			rotation = rotation - 180
		t.set_rotation(rotation)
	
	# Legend 
	ax.legend(labels, loc='upper left',
		   fontsize = fontsize_legend)
	h,l = ax.get_legend_handles_labels()
	# Equal aspect ratio ensures that pie is drawn as a circle
	ax.axis('equal')
	
	## Second pie
	
	# If the second pie flag is on, a second pie is plotted, where the data
	# are plotted for different types of players.
	if second_pie_flag is True:
		# Data handling - data grouped by type are sorted
		newdata = data.groupby(second_group_col).agg('sum').reset_index()
		newdata.loc[:, second_group_col] = \
			 pd.Categorical(newdata[second_group_col],
				   categories=second_group_labels)
		newdata.sort_values(by=[second_group_col, values_col],
				   ascending=[True, False],
				   inplace=True)
		n_slices = len(newdata)
		
		# Colormap to be used for the slices of the pie
		cmap_name = 'winter'
		cmap = plt.get_cmap(cmap_name)
		colors = [cmap(i) for i in np.linspace(0, 1, n_slices)]
		
		# A second ax is generated (y axis, right) to plot the second pie
		axtw = ax.twinx()
		labels = list(newdata[second_group_col])
		labels = [label.title() for label in labels]
		w, l, p = \
			axtw.pie(newdata[values_col], labels=labels,
			startangle=0, colors=colors, radius=0.5, 
			autopct='%1.1f%%', pctdistance=0.8, labeldistance=None,
			frame=False, shadow=False,
			textprops = {'fontsize':fontsize_pielabels}, 
			wedgeprops={'edgecolor': 'k','linewidth': 1, 'linestyle': '-.'})
		
		# Each pct text is rotated in order to be radially printed, with
		# variable distance from the centre
		pctdists = np.linspace(1, 1, n_slices)
		for t, d in zip(p, pctdists):
			# Position of the text and distance from the centre
			xi,yi = t.get_position()
			ri = np.sqrt(xi**2+yi**2)
			# Angle (rad)
			phi = np.arctan2(yi,xi)
			# New distance from the centre, selected from pctdists
			x = d*ri*np.cos(phi)
			y = d*ri*np.sin(phi)
			t.set_position((x,y))
			# Rotation of the text, radially rotated (°)
			rotation = phi/np.pi * 180
			tol = 1e-1
			if (rotation-90 >= -tol) or (rotation+90 <= tol):
				rotation = rotation - 180
			t.set_rotation(rotation)
		
		# Legend 
		axtw.legend(labels, loc='upper right',
			  fontsize = fontsize_legend)
	
	# Centre circle to turn pies into donuts
	if second_pie_flag is True: radius = 0.2
	else: radius = 0.4
	centre_circle = \
		plt.Circle((0,0), radius, color='black', fc='white', linewidth=1)
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)

	## Return
	return fig



# =============================================================================
# data = [0.002, 0.011, 0.029, 0.733, 0.027, 0.198]
# labels = ['elementari', 'incontro', 'materna', 'municipio', 'palestra', 'rsa']
# types = ['consumer', 'consumer', 'consumer', 'prosumer', 'consumer', 'consumer']
# # =============================================================================
# # data = [0.05]*20
# # labels = [str(i) for i in range(20)]
# # types = ['producer']*1 + ['prosumer']*4 + ['consumer']*15
# # =============================================================================
# 
# plot_params = {}
# fig_specs = {'suptitle': ""}
# data = pd.DataFrame({'player': labels, 'share': data, 'type': types})
# fig = \
# 	shares_pie_plot(data, second_pie_flag=True, fig_specs=fig_specs,)
# =============================================================================



