import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils.random import sample_without_replacement
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy
import matplotlib.colors as colors
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.curdir)


def visualize2D(pop, pop_acc, query, seed=0, sample_num=10000, acc_dyn=(0.88, 0.95)):
    ## load embedding
    out_path = '/home/liugroup/ranker/AG-Net-main/search/'
    # feature = []
    # test_acc = []
    feature = [x.detach().numpy() for x in pop]
    test_acc = [acc.detach().numpy() for acc in pop_acc]
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0).flatten()
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    ## architecture density
    fig, ax = plt.subplots(figsize=(5, 5))
    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)
    H, xedges, yedges, img = ax.hist2d(emb_x, emb_y, bins=(xedges, yedges))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join('/home/liugroup/ranker/AG-Net-main/search/', 'density-{}-{} points.png'.format(query, sample_num)),
                bbox_inches='tight')
    plt.close()


    # accuracy distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    acc = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                acc[xx, yy] = np.mean(test_acc[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_acc = np.ma.masked_where(acc == 0, acc)
    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    ## raw version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_raw.png'.format(query, sample_num)),
                bbox_inches='tight')
    plt.close()

    ## smooth version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   interpolation='bilinear',
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_smooth.png'.format(query, sample_num)),
                bbox_inches='tight')
    plt.close()

    ## scatter version
    x1 = emb_x[test_acc >= 0.94]
    y1 = emb_y[test_acc >= 0.94]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.scatter(emb_x, emb_y, c=test_acc, s=1, cmap=palette, norm=colors.Normalize(vmin=0.88, vmax=0.95))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax.scatter(x1, y1, c='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_scatter.png'.format(query, sample_num)),
                bbox_inches='tight')
    plt.close()