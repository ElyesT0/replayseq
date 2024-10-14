import matplotlib.pyplot as plt
import numpy as np
def pretty_gat(score,train_times,test_times,chance):
    fig, ax = plt.subplots(1, 1,figsize=(100,10))
    im = ax.imshow(
        score,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=[test_times[0],test_times[-1],train_times[0],train_times[-1]],
        vmin=chance-2*np.std(score),
        vmax=chance+2*np.std(score),
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("score")

    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# _________________________________________________________________________________________________________
def plot_category_decoding_proportions(y_true, predicted_matrix):
    """
    predicted_matrix = y_preds_diag
    Plots the proportion of times each category was decoded, split by the true category presented.

    Parameters:
    true_categories (list or ndarray): A list or array of true category labels (length N).
    predicted_matrix (ndarray): A matrix of predicted category labels (shape: N x T),
                                where N is the number of samples and T is the number of time points.
    """
    num_categories = len(np.unique(y_true))
    y_true[y_true==99] = num_categories-1
    time_steps = predicted_matrix.shape[1]

    # Create subplots
    fig, axes = plt.subplots(num_categories, 1, figsize=(10, num_categories * 3))
    fig.tight_layout(pad=3.0)

    # considered_category labels each of the 7 subplots
    for considered_category in range(num_categories):
        # find the trials where considered_category was presented
        considered_cat_indices = np.where(y_true == considered_category)[0]

        # determine the proportion of times each of the 7 categories were predicted when it was considered_category that was presented
        for cat in range(num_categories):
            decoded_proportions = np.mean(predicted_matrix[considered_cat_indices] == cat, axis=0)
            axes[considered_category].plot(range(time_steps), decoded_proportions, label=f'proportion of predicted category {cat}')

        # Set labels and titles for each subplot
        axes[considered_category].set_title(f'When category {considered_category} was shown, proportion of times all of them were predicted',cmap='viridis')
        axes[considered_category].set_xlabel('Time Steps')
        axes[considered_category].set_ylabel('Proportion')
        axes[considered_category].legend()

        # change the color to viridis map



    # Show the plot
    plt.show()



def pretty_decod(scores, times=None, chance=0, ax=None, sig=None, width=3.,
                 color='k', fill=False, xlabel='Time', sfreq=250, alpha=.75):
    scores = np.array(scores)

    if (scores.ndim == 1) or (scores.shape[1] <= 1):
        scores = scores[:, None].T
    if times is None:
        times = np.arange(scores.shape[1]) / float(sfreq)

    # setup plot
    if ax is None:
        ax = plt.gca()

    # Plot SEM
    if scores.ndim == 2:
        scores_m = np.nanmean(scores, axis=0)
        n = len(scores)
        n -= sum(np.isnan(np.mean(scores, axis=1)))
        sem = np.nanstd(scores, axis=0) / np.sqrt(n)
        plot_sem(times, scores, color=color, ax=ax)
    else:
        scores_m = scores
        sem = np.zeros_like(scores_m)

    # Plot significance
    if sig is not None:
        sig = np.squeeze(sig)
        widths = width * sig
        if fill:
            scores_sig = (chance + (scores_m - chance) * sig)
            ax.fill_between(times, chance, scores_sig, color=color,
                            alpha=alpha, linewidth=0)
            ax.plot(times, scores_m, color='k')
            plot_widths(times, scores_m, widths, ax=ax, color='k')
        else:
            plot_widths(times, scores_m, widths, ax=ax, color=color)

    # Pretty
    ymin, ymax = min(scores_m - sem), max(scores_m + sem)
    ax.axhline(chance, linestyle='dotted', color='k', zorder=-3)
    ax.axvline(0, color='k', zorder=-3)
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([ymin, chance, ymax])
    ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax])
    xticks, xticklabels = _set_ticks(times)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    if len(xlabel):
        ax.set_xlabel(xlabel)
    pretty_plot(ax)
    return ax


def pretty_plot(ax=None):
    if ax is None:
        plt.gca()
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def _set_ticks(times):
    ticks = np.arange(min(times), max(times), .100)
    if np.round(max(times) * 10.) / 10. == max(times):
        ticks = np.append(ticks, max(times))
    ticks = np.round(ticks * 10.) / 10.
    ticklabels = ([int(ticks[0] * 1e3)] +
                  ['' for ii in ticks[1:-1]] +
                  [int(ticks[-1] * 1e3)])
    return ticks, ticklabels


def plot_widths(xs, ys, widths, ax=None, color='b', xlim=None, ylim=None,
                **kwargs):
    xs, ys, widths = np.array(xs), np.array(ys), np.array(widths)
    if not (len(xs) == len(ys) == len(widths)):
        raise ValueError('xs, ys, and widths must have identical lengths')
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)

    segmentx, segmenty = [xs[0]], [ys[0]]
    current_width = widths[0]
    for ii, (x, y, width) in enumerate(zip(xs, ys, widths)):
        segmentx.append(x)
        segmenty.append(y)
        if (width != current_width) or (ii == (len(xs) - 1)):
            ax.plot(segmentx, segmenty, linewidth=current_width, color=color,
                    **kwargs)
            segmentx, segmenty = [x], [y]
            current_width = width
    if xlim is None:
        xlim = [min(xs), max(xs)]
    if ylim is None:
        ylim = [min(ys), max(ys)]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax if fig is None else fig

def plot_sem(x, y, robust=False, **kwargs):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    robust : bool
        If False use mean + std,
        If True median + mad
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    x, y = np.array(x), np.array(y)
    if robust:
        m = np.nanmedian(y, axis=0)
        std = np.nanstd(y, axis=0)
    else:
        m = np.nanmean(y, axis=0)
        std = np.nanstd(y, axis=0)
    n = y.shape[0] - np.sum(np.isnan(y), axis=0)

    return plot_eb(x, m, std / np.sqrt(n), **kwargs)

def plot_eb(x, y, yerr, ax=None, alpha=0.3, color=None, line_args=dict(),
            err_args=dict()):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    yerr : list | np.array() | float
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    x, y = np.array(x), np.array(y)
    ax = ax if ax is not None else plt.gca()
    if 'edgecolor' not in err_args.keys():
        err_args['edgecolor'] = 'none'
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y,  color=color, **line_args)
    ax.fill_between(x, ymax, ymin, alpha=alpha, color=color, **err_args)

    return ax