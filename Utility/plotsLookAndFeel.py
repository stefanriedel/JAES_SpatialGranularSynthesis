""" Style File with variables used in all plot scrtips """


import matplotlib
cmap = matplotlib.cm.get_cmap('cool')
colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8), 'k']

leg_mkr = 'D'
lev_mkr = 'o'

#mkr_color_2d = 'k'
#mkr_color_3d = 'white'

mkr_color_2d = colors[0]
mkr_color_3d = colors[3]

mkr_color_lev = colors[0]
mkr_color_leg = colors[3]

mkr_size = 12

lev_linest = '--'
leg_linest = ':'

lowpass_linest = '-.'
lowpass_color = 'k'
broadband_color = colors[2]

fs_labels = 14

linecolor = 'k'