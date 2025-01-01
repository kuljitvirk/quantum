from __future__ import print_function
import os
import warnings

import matplotlib as mpl

from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Rectangle
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

try:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
except:
    colors = None
try:
    import seaborn as sns
except:
    pass

pjoin = os.path.join
golden_ratio = (1. + np.sqrt(5))/2.
VERBOSE = False
SAVEDIR = '.'

show = plt.show
def subplots(*args, xlabel=None,ylabel=None,title=None,**kwargs):
    aspect = kwargs.pop('aspect',None)
    fig,ax = plt.subplots(*args, facecolor='w', **kwargs)
    if aspect:
        ax.set_aspect(aspect)
    if xlabel or ylabel:
        if isinstance(ax,np.ndarray):
            axes = ax
        else:
            axes = [ax]
        for ax1 in axes:
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)       
    return fig,ax

def stylefile(filename):
    if not os.path.isfile(filename):
        filename = os.path.join(os.path.dirname(__file__),filename)
    if os.path.isfile(filename):
        plt.style.use(filename)
    else:
        raise FileNotFoundError(filename)
    return


def twinx(ax,clr_ax,clr_new,ylabel=''):
    ax.spines['right'].set_color(clr_ax)
    ax.tick_params(axis='y', colors=clr_ax)
    ax2 = ax.twinx()
    ax2.set_ylabel(ylabel,color=clr_new)
    ax2.spines['right'].set_color(clr_new)
    ax2.tick_params(axis='y', colors=clr_new)
    return ax2

def axesoff(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
def axeson(ax):
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    return ax
def scientificformat(ax,axis):
    ax.ticklabel_format(axis=axis,scilimits=(0,0))
    return ax

def tickfontsize(ax,fontsize):
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    return ax

def drawbox(ax,center,size,fill=None,edgecolor='k'):
    ax.add_patch(Rectangle((center[0]-size[0]/2,center[1]-size[1]/2),size[0],size[1],fill=fill,edgecolor=edgecolor))

def savepng(figobj,figname,figsize=None,**kwargs):
    if figname[:-4:] != '.png': figname = figname+'.png'
    if figsize is not None:
        figobj.set_figwidth(figsize[0])
        figobj.set_figheight(figsize[1])
    figobj.savefig(figname, format='png',
        dpi=300,bbox_inches='tight', **kwargs)
    if VERBOSE: print('Saved:',figname)
    return

def savepdf(figobj,figname,figsize=None,**kwargs):
    if figname[:-4:] != '.pdf': figname = figname+'.pdf'
    if figsize is not None:
        figobj.set_figwidth(figsize[0])
        figobj.set_figheight(figsize[1])
    figobj.savefig(figname, format='pdf',
        dpi=300,bbox_inches='tight', **kwargs)
    if VERBOSE: print('Saved:',figname)
    return

def plot_contourf(*args,**kwargs):
    plt.contourf(*args,**kwargs)
    plt.gca().set_aspect('equal')
    return

def newfigure(**kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    return fig,ax
def setfontsize(ax,fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    return ax

def genplot(
    *args,
    figsize=None,
    dpi=100,
    yscale='linear',
    xscale='linear',
    aspect=None,
    xlabel=None,
    ylabel=None,
    title=None,
    grid=False,
    remove_spines=None,
    **kwargs):
    """
    """
    fig,ax=plt.subplots(figsize=figsize,dpi=100,facecolor='w')
    ax.plot(*args,**kwargs)
    ax.grid(grid)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if aspect is not None:
        ax.set_aspect(aspect)
    if remove_spines is not None:
        despine(ax,remove_spines)
    return fig,ax

def myimage(A,*args,ax=None,cbar_format=None,figsize=(6,6),flipy=False,**kwargs):
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[0])
    im = ax.imshow(A,**kwargs)
    if np.abs(A.max()-A.min())<0.1:
        cbar_format='%.1e'
    else:
        cbar_format='%.2f'
    fig.colorbar(im,ax=ax, format=cbar_format)
    ax.grid(False)
    if flipy:
        ax.set_ylim(ax.get_ylim()[::-1])
    return fig,ax,im

def mypcolor(x,y,A,*args,ax=None,**kwargs):
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    im = ax.pcolormesh(x,y,A,**kwargs)
    fig.colorbar(im,ax=ax)
    ax.grid(False)
    return fig,ax
#
def pupilplot(kvec,ax=None,markersize=2,marker='o',**kwrds):
    """
    """
    if np.abs(kvec[:,-1]).max() < 1.e-12:
        print('all kz = 0, kvectors will fall on NA=1 this way. Specify wavelength or kz=kvec[:,2]')
    kmag = np.sqrt((kvec**2).sum(axis=-1))
    sel = kmag>1.e-12
    kmag1 = kmag.copy()
    kmag = kmag1[sel]
    if (~sel).sum()>0:
        print('pupilplot: BAD POINTS',kvec[~sel])
    kna = kvec[sel] / np.tile(kmag[:,None],(1,kvec.shape[1]))
    if ax is None:
        fig,ax=genplot()
    else:
        fig = ax.get_figure()
    t = np.linspace(0.,2*np.pi,100)
    ax.plot(kna[:,0],kna[:,1],linewidth=0,marker=marker,markersize=markersize,**kwrds)
    ax.plot(np.cos(t),np.sin(t),'-k',label='NA = 1')
    NA = 0.9
    ax.plot(NA*np.cos(t),NA*np.sin(t),'--b',label='NA = %.2f'%(NA))
    ax.set_aspect('equal')
    wavelength = 2*np.pi/kmag[0]
    ax.set_title('Wavelength = %.0f, NA = %.2f, Nk = %d'%(wavelength, NA,len(kna)),fontsize=7)
    #ax.legend() #bbox_to_anchor=(1.05,0.9))
    return fig,ax
#
def fpr_plot_bb(ax,table,alpha=0.5,cmap='Blues'):
    """
    ax can be None
    table is pandas table
    """
    # Bare Bones
    nfpr = 0
    for i, row in table.iterrows():
        nfpr = max(nfpr,len(row.fpr_vals))
    FPRpos = np.zeros((len(table),nfpr)) + np.nan
    FPRneg = np.zeros((len(table),nfpr)) + np.nan
    Ypos = np.zeros((len(table), nfpr), dtype=float)
    Yneg = np.zeros((len(table), nfpr), dtype=float)
    i = -1
    for _, row in table.iterrows():
        i +=1
        I = row.intensity_at_fpr
        rng = np.arange(nfpr - len(I[0]),nfpr)
        Ypos[i, rng] = I[0]
        FPRpos[i,rng] = row.fpr_vals
        rng = np.arange(nfpr - len(I[1]),nfpr)
        Yneg[i, rng] = I[1]
        FPRneg[i,rng] = row.fpr_vals

    # Create 0 Intensity level
    Ypos = np.append(Ypos, np.zeros((Ypos.shape[0],1)), axis=1)
    Yneg = np.append(Yneg, np.zeros((Yneg.shape[0],1)), axis=1)
    W = np.tile(table.wavelength.values[:,None],(1,Yneg.shape[1]))
    Z = FPRpos
    Z = np.append(Z, np.ones((Z.shape[0],1)), axis=1)
    levels = np.arange(-nfpr, 1)
    im=[
        ax.contourf(W,Ypos,np.log10(Z),levels,alpha=alpha,cmap=cmap),
        ax.contourf(W,Yneg,np.log10(Z),levels,alpha=alpha,cmap=cmap)]
    Z = FPRneg
    Z = np.append(Z, np.ones((Z.shape[0],1)), axis=1)
    #sns.despine(bottom=True,left=True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax,im
#
def despine(ax,sides=None):
    if sides is None:
        sides=['right','top']
    elif isinstance(sides,str) and sides.lower()=='all':
        sides = ['left','right','bottom','top']
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax
def setproperty(ax,**mdict):
    if not isinstance(ax,list):
        axlist = [ax]
    else:
        axlist = ax
    for ax1 in axlist:
        for key, val in mdict.items():
            if key=='grid':
                ax1.grid(val)
            if key=='xlabel':
                ax1.set_xlabel(val)
            if key=='ylabel':
                ax1.set_ylabel(val)
    return ax
#
def image_pair(f1,f2,cmap='jet',figsize=(12,4),axs=None, **imshow_kwargs):
    if axs is None:
        fig,axs = subplots(ncols=2,figsize=figsize)
    fig = axs[0].get_figure()
    vmin = np.min([np.min(f) for f in [f1,f2]])
    vmax = np.max([np.max(f) for f in [f1,f2]])
    im=[
        axs[0].imshow(f1,cmap=cmap,vmin=vmin,vmax=vmax,**imshow_kwargs),
        axs[1].imshow(f2,cmap=cmap,vmin=vmin,vmax=vmax,**imshow_kwargs)]
    fig.colorbar(im[0], ax=axs, orientation='vertical')
    for ax1 in axs:
        ax1.set_aspect('equal')
    return fig,axs
#
def plot_signal(ax,signalTable,defects,dataset='signal_at_best_focus',pol='H',normalize_to_max=False ,figsize=(7,4),**kwargs):
    #
    import seaborn as sns
    sns.set(context='paper')
    P = signalTable.groupby('illu_polarization').get_group(pol)
    for name, tdef in P.groupby('defect'):
        if not any([d in name for d in defects]):
            continue
        isort = np.argsort(tdef.wavelength.values)
        if normalize_to_max:
            norm = np.abs(tdef.signal_at_best_focus).max()            
        else:
            norm=1
        #psf = (tdef.wavelength.values[isort]*1e-9/2)**2
        #fac = Nphotons * 6.2 * 1.602e-19 / (6.626e-34 * 3e8 / tdef.wavelength.values[isort]/1e-9)
        fac = 1/norm
        ax.plot(tdef.wavelength.values[isort], (tdef[dataset].values[isort]*fac), label=name,**kwargs)
    #
    ax.legend()
    ax.grid(False)
    sns.despine()
    # ax.set_xlabel('Wavelength [nm]')
    # ax.set_ylabel('Intensity')
    # title=pol+'-pol Far Field Signal'
    return ax.get_figure(),ax
def plot_image_output(image_data,focus_index,ax=None):
    if ax is None:
        fig,ax=genplot()
    else:
        fig=ax.get_figure()
    resize=eval(image_data['reshape_evalstr'])
    ax.imshow(resize(image_data['image_dif'][focus_index]))
    return
def get_plotly_layout(width):
    import plotly.graph_objects as go
    layout = go.Layout(
        width=width,
        margin=go.layout.Margin(t=0,b=0),
        xaxis={'constrain' : 'domain'},
        yaxis={'scaleanchor' : 'x','scaleratio' : 1,'constrain' : 'domain'})
    return layout

#
def gridplot1(
    data,
    rows_cols,
    fignum=None,
    figsize=None,
    cmap='gray',
    axeson=False,
    vmin=None,
    vmax=None,
    share_all=True,
    cbar_mode=None, # {"each", "single", "edge", None }, default: None
    cbar_location='right', # left, right,bottom,top
    cbar_format = None,
    cbar_pad : float = None,
    cbar_size="5%",
    cbar_set_cax=True, # If True, each axes in the grid has a cax attribute that is bound to associated cbar_axes.
    axes_class=None,
    axes_pad=0.1,
    spines = 'default'
    ):
    if figsize is None:
        imax = np.argmax(rows_cols)
        if imax==0:
            # rows = height, cols=width
            figsize=(rows_cols[1]/rows_cols[0]*8,8)
        else:
            figsize=(8,rows_cols[0]/rows_cols[1]*8)
    #
    fig = plt.figure(num=fignum,figsize=figsize,facecolor='w')
    fig.clear()
    b = spines in ('default','all')
    a = spines  == 'all'

    with plt.rc_context(rc={'axes.spines.top':a,'axes.spines.right':a,'axes.spines.left':b,'axes.spines.bottom':b}):
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=rows_cols,
            share_all=share_all,
            axes_pad=axes_pad,
            cbar_mode=cbar_mode,
            cbar_location=cbar_location,
            cbar_pad = cbar_pad,
            cbar_size=cbar_size,
            cbar_set_cax=cbar_set_cax,
            axes_class=axes_class
            )
    vlim=[]
    im=None
    for i in range(min(len(data), len(grid))):
        _vmin = data[i].astype(float).min() if vmin is None else vmin
        _vmax = data[i].astype(float).max() if vmax is None else vmax
        vlim += [[_vmin,_vmax]]
        im=grid[i].imshow(data[i],cmap=cmap,vmin=_vmin,vmax=_vmax)
        if cbar_mode is not None and cbar_mode=='each':
            cbar = fig.colorbar(im,ax=grid[i],location=cbar_location)
            grid[i].cbar = cbar
    if cbar_mode is not None  and im is not None and cbar_mode=='single':
        #cbar = fig.colorbar(im,ax=grid,location=cbar_location)
        if cbar_location == 'right':
            cbar = fig.colorbar(im,ax=grid[2], anchor=(1.3,0.5))
        grid.cbar = cbar
    for g in grid:
        g.get_xaxis().set_visible(axeson)
        g.get_yaxis().set_visible(axeson)
    return grid

def gridplot_focus_scans(mdict,row_col,figsize=None):
    D = np.asarray([mdict['image_dif'][i].reshape(mdict['image_shape']) for i in range(len(mdict['image_dif']))])
    vmax = max(-np.min(D),np.max(D))
    vmin = -vmax
    grid=gridplot(D, row_col, axeson=False, figsize=figsize, share_all=True, cmap='rocket',cbar_mode='none',vmin=vmin,vmax=vmax)
    for i in range(len(D)):
        grid[i].text(3,D.shape[0]-3,'{:.0f}'.format(mdict['focus_offsets'][i]),color='cyan')
    return grid
#
def colorbar(im,ax,size='5%',pad='2%',location='right',**kwargs):
    ax_divider = make_axes_locatable(ax)
    if location in ('left','right'):
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    if orientation=='horizontal':
        cax = ax_divider.append_axes('bottom',size='7%',pad='2%')
        kwargs['ticklocation']='bottom'
    cax = ax_divider.append_axes(location, size=size, pad=pad)
    cbar = ax.get_figure().colorbar(im, cax=cax, ax=ax, **kwargs)
    return cbar

def save_pcolormesh_animation(x,y,stateArray,filename,cmap='RdBu_r',cbar=False, symmetric_scale=True):
    from matplotlib.animation import PillowWriter as Writer
    writer = Writer()
    fig,ax = plt.subplots(figsize=(6,6))
    vmin = stateArray.min()
    vmax = stateArray.max()
    if symmetric_scale:
        vmin = min(vmin, -vmax)
        vmax = -vmin
    with writer.saving(fig, filename, 120):
        for state in stateArray:
            ax.cla()
            im=ax.pcolormesh(x,y,state.reshape((len(x),len(x))),cmap=cmap, shading='auto',vmin=vmin,vmax=vmax)
            axesoff(ax)
            if cbar:
                colorbar(im,ax)
            writer.grab_frame()
    #
    print('Saved: ',os.path.abspath(filename))
    return

def edge_color(ax,edges,points, Q, vmin=None,vmax=None,ncolors=100,cmap='RdBu_r',cbar=True, facecolor='lightgray',orientation='vertical'):
    colors=sns.color_palette(cmap,n_colors=1+ncolors)
    vmin = Q.min() if vmin is None else vmin
    vmax = Q.max() if vmax is None else vmax
    Qd = np.digitize(Q,np.linspace(vmin,vmax, ncolors), right=True )
    L = points
    for k,(i,j) in enumerate(edges):
        ax.plot( L[[i,j],0], L[[i,j],1] , color=colors[Qd[k]]  )
    ax.set_aspect('equal')
    ax.set_facecolor(facecolor)
    
    if cbar:
        fig = ax.get_figure()    
        fig.subplots_adjust(right=0.9)
        ax_divider = make_axes_locatable(ax)
        if orientation=='horizontal':
            cax = ax_divider.append_axes('bottom',size='7%',pad='2%')
            tloc = 'bottom'
        else:
            cax = ax_divider.append_axes('right',size='7%',pad='2%')
            tloc = 'right'
        cmap = sns.color_palette(cmap,n_colors=10,as_cmap=True)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,ticklocation=tloc,orientation=orientation)
    return

def gridplot(
    Images,
    x = None,
    y = None,
    axis=0,
    ncols = None,
    titles=None,
    axes_pad=0.1,
    figsize=(8,None),
    dpi=100,
    vmin=None,
    vmax=None,
    sharex=True,
    sharey=True,
    aspect='equal',
    cmap='viridis',
    cbar_ticklabel_size=9,
    cbar_ticklabel_rotation=0,
    cbar_mode="single",
    cbar_pad=0.02,
    cbar_location="right",
    cbar_size = 0.03):
    """
    """
    if axis==-1:
        Images = Images.transpose((-1,0,1))
    n = len(Images)
    if ncols is None:
        ncols = n

    nrows = int(n/ncols)
    if nrows*ncols < n:
        nrows += 1

    fw, fh = figsize
    fh = fw*(len(Images))/ncols**2 if fh is None else fh
    figsize = (fw,fh)

    fig,grid_ = plt.subplots(figsize=figsize,dpi=dpi,ncols=ncols,nrows=nrows,sharex=sharex,sharey=sharey)
    m = 0
    grid = grid_.flat
    if vmin is None or np.isscalar(vmin):
        vmin = [vmin]*len(grid)
    if vmax is None or np.isscalar(vmin):
        vmax = [vmax]*len(grid)

    if cbar_mode!='each':
        vmin = [np.min([I.min() for I in Images])]*len(grid)
        vmax = [np.max([I.max() for I in Images])]*len(grid)
    for ax in grid:
        axesoff(ax)
    for ax in grid[:n]:
        axesoff(ax)
        ax.set_aspect(aspect)        
        # Inside the loop, allows for a list of unequal size images
        x_ = np.arange(Images[m].shape[1]) if x is None else x
        y_ = np.arange(Images[m].shape[0]) if y is None else y
        im = ax.pcolormesh(x_,y_,Images[m],cmap=cmap,vmin=vmin[m],vmax=vmax[m])
        if cbar_mode=='each':
            cbar = colorbar(im,ax=ax,pad=cbar_pad,location=cbar_location,size=cbar_size)
            #cbar.xaxis.set_tick_params(labelrotation=cbar_ticklabel_rotation,labelsize=cbar_ticklabel_size)
        m += 1
    if cbar_mode != 'each':
        b = grid[n-1].get_position()
        x0 = grid[-1].get_position().x1
        h = grid[ncols-1].get_position().y1 - b.y0
        y0 = 0.5-h/2
        cax=fig.add_axes([x0+cbar_pad,y0,cbar_size,h])
        cbar = fig.colorbar(im, cax=cax, fraction=1,pad=0)
        # cax.set_tick_params(labelrotation=cbar_ticklabel_rotation,labelsize=cbar_ticklabel_size)
    if titles is not None and len(titles)==len(grid):
        for i, ax in enumerate(grid):
            ax.set_title(titles[i])
    for ax in grid[len(Images):]:
        ax.axis('off')
    return fig,grid_

def colorbar_gridplot(im,grid,cbar_pad=0.02,cbar_size = 0.03,format='%.1e'):
    n = grid.size
    ncols = grid.shape[1]
    fig = grid[0][0].get_figure()
    x0 = grid[-1][-1].get_position().x1
    b = grid[-1][-1].get_position()
    h = grid[0][-1].get_position().y1 - b.y0
    y0 = 0.5-h/2
    cax=fig.add_axes([x0+cbar_pad,y0,cbar_size,h])
    cbar = fig.colorbar(im, cax=cax, fraction=1,pad=0,format=format)
    return cbar

def smatrix_plot(ax,ko,ki,dS,spacing=1,cmap='RdBu_r',vmin=None,vmax=None):
    ax.set_aspect('equal')
    X = np.abs(np.diff(ki,axis=0)[:,:2])
    dk = np.min(X[X>1e-12])
    scale = 2*spacing/dk
    for i in range(len(ki)):
        _ko = scale*ki[i] + ko
        im=ax.scatter(_ko[:,0],_ko[:,1],c=dS[:,i],s=2,marker='o',cmap=cmap,vmin=vmin,vmax=vmax)
    axesoff(ax)
    return im

def smatrix_plot_pol(ko,ki,dS,spacing=1,cmap='RdBu_r',vmin=None,vmax=None,format='%.1e'):
    fig,axs_ = subplots(figsize=(8,8),dpi=100,ncols=2,nrows=2)
    axs = axs_.flat
    vmin = (dS).min()
    vmax = (dS).max()
    n = len(ki)
    kwargs = dict(cmap=cmap,vmin=vmin,vmax=vmax,spacing=spacing)
    for i,ds in enumerate([dS[:n,:n],dS[:n,n:],dS[n:,:n],dS[n:,n:]]):
        im=smatrix_plot(axs[i], ko,ki,ds,**kwargs)
    colorbar_gridplot(im,axs_,format=format)
    for ax in axs_.flat:
        ax.axis('off')
    return fig,axs_
