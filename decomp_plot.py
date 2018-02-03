import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def separate_transform_result(D,ori_data,ping_per_day_mvbs,log_opt=1):
    '''
    Separate transformed results into different frequencies and 
    for use with `plot_cmp_data_decomp` and `plot_single_day`
    '''
    D_long = D.reshape((D.shape[0],-1,ori_data.shape[1])).swapaxes(1,2)
    D_sep = D_long.reshape((D_long.shape[0],D_long.shape[1],-1,ping_per_day_mvbs)).transpose((2,0,1,3))
    if log_opt==1:
        D_plot = 10*np.log10(D_sep.transpose((0,2,1,3))).reshape((D_sep.shape[0],D_sep.shape[2],-1))
    else:
        D_plot = D_sep.transpose((0,2,1,3)).reshape((D_sep.shape[0],D_sep.shape[2],-1))
    return D_sep,D_plot



def plot_single_day(V,plot_day,ping_per_day_mvbs):
    fig,ax = plt.subplots(1,3,figsize=(18,3))
    
    # Get color axis limtis
    v_mtx = V[:,1:-2,ping_per_day_mvbs*(plot_day-1)+np.arange(ping_per_day_mvbs)]  # don't plot surface/bottom rows
    cmean = np.mean(v_mtx.reshape((-1,1)))
    cstd = np.std(v_mtx.reshape((-1,1)))
    cmax = np.max(v_mtx.reshape((-1,1)))

    for iX in range(3):
        im = ax[iX].imshow(v_mtx[iX,::-1,:],aspect='auto',vmax=cmean+cstd*6,vmin=cmean-cstd*3)#,cmap=e_cmap,norm=e_norm)
        divider = make_axes_locatable(ax[iX])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im,cax=cax)
        if iX==0:
            ax[iX].set_title('38 kHz')
        elif iX==1:
            ax[iX].set_title('120 kHz')
        else:
            ax[iX].set_title('200 kHz')
    #plt.savefig(os.path.join(save_path,save_fname))    
    


def plot_comp(V,n_comp,ping_per_day_mvbs,figsize_input,log_opt=1,cax_all=0,cax=np.nan):
    if log_opt==1:
        V = 10*np.ma.log10(V)
    if np.any(np.isnan(cax)):
        cmean_all = np.mean(V)
        cstd_all = np.std(V)
        cmin_all = max((np.min(V),cmean_all-2*cstd_all))
        cmax_all = min((np.max(V),cmean_all+3*cstd_all))
    else:
        cmin_all = cax[0]
        cmax_all = cax[1]

    fig,ax=plt.subplots(n_comp,1,sharex=True,figsize=figsize_input)
    for c in range(n_comp):
        if log_opt==1:
            vlog = 10*np.ma.log10(V[c,:,:])
        else:
            vlog = V[c,:,:]
        cmean = np.mean(V[c,:,:])
        cstd = np.std(V[c,:,:])
        if cax_all==1:
            cmin = cmin_all
            cmax = cmax_all
        else:
            cmin = max((np.min(V[c,:,:]),cmean-2*cstd))
            cmax = min((np.max(V[c,:,:]),cmean+3*cstd))
        im = ax[c].imshow(V[c,::-1,:],aspect='auto',vmin=cmin,vmax=cmax)
        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im,cax=cax)

        ax[c].set_xticks([x*ping_per_day_mvbs+ping_per_day_mvbs/2 for x in range(3)])
        ax[c].set_xticklabels(['38k','120k','200k'])
        ax[c].tick_params('both', length=0)
    #plt.savefig(os.path.join(save_path,save_fname))
    
    
    
def plot_coef(W,n_comp,figsize_input=(22,3),log_opt=0):
    plt.figure(figsize=figsize_input)
    W[W==0] = sys.float_info.epsilon
    labels = [str(x) for x in range(n_comp)]
    for w, label in zip(W.T, labels):
        plt.plot(range(1,len(w)+1),w, label=label,linewidth=2)
    plt.legend()
    plt.xticks(range(W.shape[0]))
    if log_opt==1:
        plt.yscale('log')
    plt.xlim([0,W.shape[0]])
    #plt.savefig(os.path.join(save_path,save_fname))
    plt.show()
    

        
def plot_cmp_data_decomp(V,X,plot_day,ping_per_day_mvbs,figsize_input,same_cax_opt=1):
    fig,ax = plt.subplots(2,3,figsize=figsize_input)
    for iY in range(2):
        # Get color axis limtis
        v_mtx = V[:,:,ping_per_day_mvbs*(plot_day-1)+np.arange(ping_per_day_mvbs)].reshape((-1,1))
        cmean = np.mean(v_mtx)
        cstd = np.std(v_mtx)
        cmax = np.max(v_mtx)

        for iX in range(3):
            if iY==0:
                v = V[iX,::-1,ping_per_day_mvbs*(plot_day-1)+np.arange(ping_per_day_mvbs)]   # data to be plotted
            else:
                v = X[iX,::-1,ping_per_day_mvbs*(plot_day-1)+np.arange(ping_per_day_mvbs)]   # data to be plotted

            if same_cax_opt==1:
                im = ax[iY,iX].imshow(v.T,aspect='auto',vmax=cmean+cstd*6,vmin=cmean-cstd*3)
            else:
                im = ax[iY,iX].imshow(v.T,aspect='auto')
            divider = make_axes_locatable(ax[iY,iX])
            cax = divider.append_axes("right", size="2%", pad=0.1)
            cbar = plt.colorbar(im,cax=cax)
            if iX==0:
                ax[iY,iX].set_title('38 kHz')
            elif iX==1:
                ax[iY,iX].set_title('120 kHz')
            else:
                ax[iY,iX].set_title('200 kHz')
    #plt.savefig(os.path.join(save_path,save_fname))
    
    
    
def plot_original_echogram(MVBS,plot_start_day,plot_range_day):
    fig,ax = plt.subplots(3,1,figsize=(15,6))
    ax[0].imshow(MVBS[0,1:-2:-1,ping_per_day_mvbs*(plot_start_day-1)\
                      +np.arange(ping_per_day_mvbs*plot_range_day)].T,\
                 aspect='auto',vmin=-80,vmax=-30)
    ax[1].imshow(MVBS[1,1:-2:-1,ping_per_day_mvbs*(plot_start_day-1)\
                      +np.arange(ping_per_day_mvbs*plot_range_day)].T,\
                 aspect='auto',vmin=-80,vmax=-30)
    ax[2].imshow(MVBS[2,1:-2:-1,ping_per_day_mvbs*(plot_start_day-1)\
                      +np.arange(ping_per_day_mvbs*plot_range_day)].T,\
                 aspect='auto',vmin=-80,vmax=-30)
    ax[2].set_xticks(np.arange(plot_range_day)*ping_per_day_mvbs+ping_per_day_mvbs/2)
    ax[2].set_xticklabels(np.arange(plot_range_day)+plot_start_day)
    ax[2].set_xlabel('Day',fontsize=14)
    plt.show()
    
    
    
