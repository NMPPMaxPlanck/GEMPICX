import matplotlib.pyplot as plt

def annotate_y_ticks(y_values, y_tick_labels, arrow_style=None):
    """
    Annotate multiple points on the y-axis and indicate their positions with arrows.

    Parameters:
    - y_values: A list of y-axis values to annotate (e.g., [omegalh, omegauh, omega_L, omega_R]).
    - y_tick_labels: A list of labels corresponding to the y-axis values 
                     (e.g., [r'$\omega_{lh}$', r'$\omega_{uh}$', r'$\omega_{L}$', r'$\omega_{R}$']).
    - arrow_style: A dictionary controlling the appearance of the arrows (optional, 
                   default is facecolor='black', shrink=0.05, width=0.1, headwidth=1.5).
    """
    if arrow_style is None:
        arrow_style = dict(facecolor='black', shrink=0.05, width=0.1, headwidth=1.5)
    ax = plt.gca()
    
    for y, label in zip(y_values, y_tick_labels):
        plt.annotate(label, xy=(0, y), xytext=(-0.5, y),
                     textcoords='data', arrowprops=arrow_style,
                     horizontalalignment='right', verticalalignment='center')
        # Draw a horizontal dashed line at the corresponding y-value
        ax.axhline(y, color='black', linestyle='--', linewidth=0.5, zorder=1)
