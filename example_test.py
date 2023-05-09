'''
author: Andrew Smith, 9 May 2023

Code which runs some example cases for magnet_plotter
'''

from magnet_field_plotter import magnet_system, displayer, get_filled_hexagon_circular_state, get_filled_hexagon_unstable_circular_state, get_filled_hexagon_dipolar_state, get_polygon
nx, ny = 2048, 2048
r_max = 2.5
Z_plane = 0.5

system = get_filled_hexagon_circular_state(alpha=1)
disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
disp.plot_field_lines('plots/plot_field_lines_circular_alpha=1.png')
disp.plot_field_strength_contours('plots/plot_field_strength_circular_alpha=1.png', img_underlay='source_images/453p.png', img_extent=3.37)

system = get_filled_hexagon_unstable_circular_state(alpha=1)
disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
disp.plot_field_lines('plots/plot_field_lines_circular_unstable_circular_alpha=1.png')
disp.plot_field_strength_contours('plots/plot_field_strength_unstable_circular_alpha=1.png', img_underlay='source_images/beta3.png', img_extent=3.6)

system = get_filled_hexagon_dipolar_state(alpha=2.4)
disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
disp.plot_field_lines('plots/plot_field_lines_dipolar_alpha=2.4.png')
disp.plot_field_strength_contours('plots/plot_field_strength_dipolar_alpha=2.4.png', img_underlay='source_images/592b.png', img_extent = 3.43)

system = get_polygon(n = 6)
disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
disp.plot_field_lines('plots/plot_field_lines_hex.png')
disp.plot_field_strength_contours('plots/plot_field_strength_hex.png', img_underlay='source_images/420b.png', img_extent = 3.47)

system = get_polygon(n = 5)
disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
disp.plot_field_lines('plots/plot_field_lines_pentagon.png')
disp.plot_field_strength_contours('plots/plot_field_strength_pentagon.png', img_underlay=None)
