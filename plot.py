import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import  PatchCollection
from PIL.ImageChops import offset
from jedi.api.refactoring import inline
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import lineStyles
from scipy.stats import alpha
from sklearn.linear_model import LinearRegression


# def fit_model(x_train: np.ndarray, y_train: np.ndarray):
#     regression = LinearRegression()
#     regression.fit(x_train, y_train)
#     b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
#     return b_minimum, w_minimum


def find_index(b, w, bs: np.ndarray, ws: np.ndarray):
    b_idx = np.argmin(np.abs(bs[0, :] - b))
    w_idx = np.argmin(np.abs(ws[:, 0] - w))

    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]

    return b_idx, w_idx, fixedb, fixedw


def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].scatter(x_train, y_train)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title("Generated Data - Train")

    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title("Generated Data - Validation")


def figure2(x_train, y_train, b, w, color='k'):
    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0,3])
    
    ax.scatter(x_train, y_train)

    x_range = np.linspace(0,1,101)
    yhat = b + w*x_range
    ax.plot(x_range, yhat, label='Model\'s prediction', c=color, linestyle='--')

    ax.annotate('b={:.4f} w={:.4f}'.format(b[0], w[0]), xy=(0.4,0.6))
    ax.legend(loc=0)
    # fig.tight_layout()
    return fig, ax


def figure3(x_train: np.ndarray, y_train: np.ndarray, b: np.ndarray, w: np.ndarray):
    fig, ax = figure2(x_train=x_train, y_train=y_train, b=b, w=w)
    x0, y0 = x_train[0][0], y_train[0][0]

    ax.scatter([x0], [y0], c='r')
    ax.scatter([x0], [b[0] + w[0]*x0])

    ax.plot([x0, x0], [y0, b[0] + w[0]*x0], linestyle="--", c='r')
    ax.arrow(x0, y0 - 0.05, 0, 0.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=0.03)
    ax.arrow(x0, b[0]+w[0]*x0+0.05, 0, -0.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=0.03 )
    ax.annotate(r'$error_0$', xy=(.8, 1.5))

    fig.tight_layout()


def figure4(x_train: np.ndarray, y_train: np.ndarray, b, w, bs, ws, all_losses):
    figure = plt.figure(figsize=(12,6))

    ax1 = figure.add_subplot(1,2,1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Loss surface')
    surf = ax1.plot_surface(
        bs,
        ws,
        all_losses,
        alpha=0.5,
        cmap=plt.cm.jet,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True
    )
    ax1.contour(bs[0,:], ws[:,0], all_losses, 10, offset=-1, cmap=plt.cm.jet)

    b_minimum_idx, w_minimum_idx, _, _ =  find_index(1, 2, bs, ws)
    ax1.scatter(1, 2, all_losses[w_minimum_idx, b_minimum_idx ], c='k')
    ax1.text(0, 2.5, all_losses[w_minimum_idx, b_minimum_idx], s=f"Minimum=\n{all_losses[w_minimum_idx, b_minimum_idx]}")

    bidx, widx,_,_ = find_index(b,w, bs, ws)
    ax1.scatter(xs=b,ys=w, zs=all_losses[widx,bidx], c='k')
    ax1.text(x=-1, y=-0.5, z=all_losses[widx,bidx], s="Random Start", c='k')
    ax1.view_init(40,260)

    ax2 = figure.add_subplot(1,2,2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    CS = ax2.contour(bs[0,:], ws[:,0], all_losses, cmap=plt.cm.jet)
    ax2.clabel(CS, inline=1, fontsize=10)

    ax2.scatter(x=1, y=2, c='k')
    ax2.annotate(text="Minimum", xy=(0.8,2.3))

    ax2.scatter(x=b,y=w, c='k')
    ax2.annotate(text="Random Start", xy=(0.2,0.1))

    figure.tight_layout()
    return figure, (ax1,ax2)

def figure5(b,w,bs:np.ndarray,ws:np.ndarray,all_losses:np.ndarray):
    figure = plt.figure(figsize=(8,12))

    ax1 = figure.add_subplot(2,1,1)
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    CS = ax1.contour(bs[0,:], ws[:,0], all_losses, cmap=plt.cm.jet)
    ax1.clabel(CS)
    ax1.scatter(x=b,y=w, c='k')

    b_idx, w_idx, fixedb, fixedw = find_index(b,w,bs,ws)
    ax1.plot([fixedb, fixedb], [ws[0,0],ws[-1, 0]], linestyle='--', c='r')

    ax2 = figure.add_subplot(2,1,2)
    ax2.set_xlabel('w')
    ax2.set_ylabel('loss')
    ax2.plot(ws[:,0], all_losses[:,b_idx], linestyle='--', c='r')

    ax2.scatter(x=fixedw, y=all_losses[w_idx,b_idx], c='r')
    figure.tight_layout()


def figure6(b,w,bs:np.ndarray,ws:np.ndarray,all_losses:np.ndarray):
    figure = plt.figure(figsize=(8,12))

    ax1 = figure.add_subplot(2,1,1)
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.contour(bs[0,:], ws[:,0], all_losses, cmap=plt.cm.jet)
    bidx,widx,fixedb,fixedw = find_index(b,w,bs,ws)
    ax1.scatter(x=fixedb, y=fixedw, c='k')
    ax1.text(x=fixedb+0.1, y=fixedw+0.1, s="Random start")
    ax1.plot([bs[0,0], bs[0,-1]], [fixedw, fixedw], linestyle='--', c='r')

    ax2 = figure.add_subplot(2,1,2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('loss')
    ax2.plot(bs[0,:], all_losses[widx,:], linestyle='--', linewidth=2)
    ax2.scatter(x=fixedb, y=all_losses[widx,bidx], c='k')

    figure.tight_layout()




def figure7(b, w, bs:np.ndarray, ws:np.ndarray, all_loss:np.ndarray):
    figure = plt.figure(figsize=(12,6))

    bidx, widx, fixedb, fixedw = find_index(b,w,bs,ws)

    ax1 = figure.add_subplot(1,2,1)
    ax1.set_title(f"Fixed b={fixedb}")

    ax1.set_xlabel('w')
    ax1.set_ylabel('loss')
    ax1.set_ylim([-1,6.1])

    ax1.plot(ws[:,0], all_loss[:,bidx])
    ax1.scatter(x=fixedw,y=all_loss[widx,bidx])
    rect = Rectangle((fixedw, all_loss[widx,bidx]-0.5), 0.5,0.5)
    pc = PatchCollection([rect], facecolors='r', alpha=0.3, edgecolors='r')
    ax1.add_collection(pc)

    ax2 = figure.add_subplot(1,2,2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('loss')
    ax2.set_title(f"Fixed w={fixedw}")
    ax2.set_ylim([-1,6.1])
    ax2.plot(bs[0,:], all_loss[widx,:])
    ax2.scatter(x=fixedb, y=all_loss[widx,bidx])
    react = Rectangle(xy=(fixedb,all_loss[widx,bidx]-0.5), width=0.5, height=0.5)
    pc = PatchCollection([react], facecolors='g', alpha=0.3, edgecolors='r')
    ax2.add_collection(pc)

    figure.tight_layout()


def calc_gradient(param_before, param_after, loss_before, loss_after):
    delta_param = param_after - param_before
    delta_loss = loss_after - loss_before
    manual_grad = delta_loss/delta_param

    return delta_param, delta_loss, manual_grad
4
def figure8(b, w, b_after, w_after, bs: np.ndarray, ws: np.ndarray, all_losses: np.ndarray):
    bidx_before, widx_before, fixedb_before, fixedw_before =  find_index(b,w,bs,ws)
    bidx_after, widx_after, fixedb_after, fixedw_after = find_index(b_after, w_after, bs, ws)
    loss_before = all_losses[widx_before, bidx_before]

    figure = plt.figure(figsize=(12,6))
    ax1 = figure.add_subplot(1,2,1)
    ax1.set_xlabel('w')
    ax1.set_ylabel('loss')






















