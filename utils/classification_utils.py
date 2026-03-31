import os
import glob
import pickle
import numpy as np
from PyPDF2 import PdfWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skewnorm

def sigmoid(x):
    x = np.float64(x)
    return expit(x)

def compute_efficiency_rejection(delta_log_likelihood, true_labels):
    from scipy.integrate import trapezoid, simpson
    thresholds = np.linspace(-20.0, 20.0, 20000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = delta_log_likelihood > thresholds_broadcasted

    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  
    rejections = TN / (TN + FP)  
    auc = simpson(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc

def fit_skewnorm(dll_k, dll_p, bins=200, normalized=True, n_bootstrap=100):
    if not normalized:
        raise ValueError("Skew-normal fit requires normalized histogram (PDF). Set normalized=True.")

    def fit_one(dll, bins):
        hist, bin_edges = np.histogram(dll, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        a_guess, loc_guess, scale_guess = 0, np.mean(dll), np.std(dll)

        popt, pcov = curve_fit(
            lambda x, a, loc, scale: skewnorm.pdf(x, a, loc, scale),
            bin_centers, hist,
            p0=[a_guess, loc_guess, scale_guess],
            maxfev=10000
        )
        a, loc, scale = popt
        delta = a / np.sqrt(1 + a**2)
        mean = loc + scale * delta * np.sqrt(2 / np.pi)
        stddev = scale * np.sqrt(1 - (2 * delta**2) / np.pi)
        return popt, mean, stddev, bin_centers

    try:
        popt_k, mean_k, stddev_k, bin_centers_k = fit_one(dll_k, bins)
        popt_p, mean_p, stddev_p, bin_centers_p = fit_one(dll_p, bins)
    except RuntimeError as e:
        print("Skewnorm fit error:", e)
        exit()

    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p) / 2.)

    # Bootstrap for uncertainty
    sigma_samples = []
    for _ in range(n_bootstrap):
        resample_k = np.random.choice(dll_k, size=len(dll_k), replace=True)
        resample_p = np.random.choice(dll_p, size=len(dll_p), replace=True)
        try:
            _, mk, sk, _ = fit_one(resample_k, bins)
            _, mp, sp, _ = fit_one(resample_p, bins)
            sigma_boot = (mk - mp) / ((sk + sp) / 2.)
            sigma_samples.append(sigma_boot)
        except RuntimeError:
            continue  # skip failed fits for now - doesn't occur

    sigma_err = np.std(sigma_samples)

    return popt_k, popt_p, sigma_sep, bin_centers_k, bin_centers_p, sigma_err, normalized

def perform_fit(dll_k,dll_p,bins=200,normalized=False):
    if normalized:
        gaussian = gaussian_normalized
    else:
        gaussian = gaussian_unnormalized

    hist_k, bin_edges_k = np.histogram(dll_k, bins=bins, density=normalized)
    bin_centers_k = (bin_edges_k[:-1] + bin_edges_k[1:]) / 2
    try:
        popt_k, pcov_k = curve_fit(gaussian, bin_centers_k, hist_k, p0=[1, np.mean(dll_k), np.std(dll_k)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_k, mean_k, stddev_k = popt_k
        perr_k = np.sqrt(np.diag(pcov_k))
    except RuntimeError as e:
        print('Kaon error, exiting.')
        print(e)
        exit()
        

    hist_p, bin_edges_p = np.histogram(dll_p, bins=bins, density=normalized)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
    try:
        popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p), np.std(dll_p)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_p, mean_p, stddev_p = popt_p
        perr_p = np.sqrt(np.diag(pcov_p))
    except RuntimeError as e:
        print('Pion error, exiting.')
        print(e)
        exit()
    
    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.)
    sigma_err = (2*perr_k[1]/(stddev_k + stddev_p))** 2 + (2*perr_p[1]/(stddev_k + stddev_p))** 2 + (-2*(mean_k - mean_p) * perr_k[2] / (stddev_k + stddev_p)**2)**2 + (-2*(mean_k - mean_p) * perr_p[2] / (stddev_k + stddev_p)**2)**2
    return popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p,np.sqrt(sigma_err), normalized

def gaussian_normalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) / (np.sqrt(2 * np.pi) * stddev)

def gaussian_unnormalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def extract_values(file_path):
    results = np.load(file_path,allow_pickle=True)
    sigmas = []
    thetas = []
    for theta, gr_value in results.items():
        if theta == 25.0:
            continue

        thetas.append(float(theta))
        sigmas.append(float(gr_value))
        
    sorted_thetas, sorted_sigmas = zip(*sorted(zip(thetas, sigmas)))

    return list(sorted_sigmas), list(sorted_thetas)

def plot_skewnorm(popt, bin_centers, min_=-15,max_=15):
    a, loc, scale = popt

    x_vals = np.linspace(min_,max_, 1000)
    pdf_vals = skewnorm.pdf(x_vals, a, loc, scale)

    return x_vals,pdf_vals


def run_plotting(out_folder,momentum,model_type='Swin',skewnorm=False):

    LL_Kaon = np.load(os.path.join(out_folder,"Kaon_DLL_Results.pkl"),allow_pickle=True)
    LL_Pion = np.load(os.path.join(out_folder,"Pion_DLL_Results.pkl"),allow_pickle=True)

    kin_p = LL_Pion['Kins']
    kin_k = LL_Kaon['Kins']
    dll_p = LL_Pion['z_value']
    dll_k = LL_Kaon['z_value']
    print("NaN Checks: ",np.isnan(dll_k).sum())
    print("NaN Checks: ",np.isnan(dll_p).sum())
    dll_k = np.clip(dll_k[~np.isnan(dll_k)],-99999,99999)
    dll_p = np.clip(dll_p[~np.isnan(dll_p)],-99999,99999)
    kin_k =  kin_k[~np.isnan(dll_k)]
    kin_p = kin_p[~np.isnan(dll_p)]

    idx = np.where(kin_k[:,0] == momentum)[0]
    dll_k = dll_k[idx]
    kin_k = kin_k[idx]

    idx = np.where(kin_p[:,0] == momentum)[0]
    dll_p = dll_p[idx]
    kin_p = kin_p[idx]

    print("Pion max/min: ", dll_p.max(),dll_p.min())
    print("Kaon max/min: ",dll_k.max(),dll_k.min())

    
    bins = np.linspace(-20,20,400) 

    ### Raw DLL
    plt.hist(dll_k,bins=bins,density=True,alpha=1.0,label=r'$\mathcal{K} - $'+str(model_type),color='red',histtype='step',lw=2)
    plt.hist(dll_p,bins=bins,density=True,alpha=1.0,label=r'$\pi - $'+str(model_type),color='blue',histtype='step',lw=2)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()


    thetas = [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150]
    seps = []
    sep_err = []
    seps_cnf = []
    sep_err_cnf = []
    fig = plt.figure(figsize=(6,4))

    for theta in thetas:
        k_idx = np.where(kin_k[:,1] == theta)[0]
        p_idx = np.where(kin_p[:,1] == theta)[0]
        print("Theta: ",theta, "Pions: ",len(p_idx)," Kaons: ",len(k_idx))
        if not skewnorm:
            popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = perform_fit(dll_k[k_idx],dll_p[p_idx],bins)
        else:
            popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = fit_skewnorm(dll_k[k_idx],dll_p[p_idx],bins)
        seps.append(abs(sep_NF))
        sep_err.append(se)

        

        if not skewnorm:
            if normalized:
                gaussian = gaussian_normalized
            else:
                gaussian = gaussian_unnormalized
            plt.plot(bin_centers_k_NF, gaussian(bin_centers_k_NF, *popt_k_NF),color='blue', label=r"$\mathcal{K}$")
            plt.plot(bin_centers_p_NF, gaussian(bin_centers_p_NF, *popt_p_NF),color='red', label=r"$\pi$")

        else:
            x_pion,pdf_pion = plot_skewnorm(popt_p_NF, bin_centers_p_NF)
            x_kaon,pdf_kaon = plot_skewnorm(popt_k_NF, bin_centers_k_NF)
            plt.plot(x_kaon, pdf_kaon,color='blue', label=r"$\mathcal{K}$")
            plt.plot(x_pion, pdf_pion,color='red', label=r"$\pi$")

        plt.hist(dll_p[p_idx],bins=bins,density=normalized,color='red',histtype='step',lw=3)
        plt.hist(dll_k[k_idx],bins=bins,density=normalized,color='blue',histtype='step',lw=3)
        plt.legend(fontsize=18) 
        plt.title(r"$\theta = $ {0}".format(theta)+ r", $\sigma = $ {0:.2f}".format(sep_NF),fontsize=18)
        plt.xlabel(r"$Ln \, L(\mathcal{K}) - Ln \, L(\pi)$",fontsize=18)
        plt.ylabel("entries [#]",fontsize=18)
        plt.savefig(os.path.join(out_folder,"Gauss_fit_theta_{0}.pdf".format(theta)),bbox_inches="tight")
        plt.close()

    seps_NF = np.load(f"../Cherenkov_FastSim/Inference/NF_Comparison/{momentum}/Separation_NF_{int(momentum)}.pkl",allow_pickle=True)
    NF_err = np.load(f"../Cherenkov_FastSim/Inference/NF_Comparison/{momentum}/Errors_NF_{int(momentum)}.pkl",allow_pickle=True)
    
    results_dict = {"sigmas": seps,"errors": sep_err}

    with open(os.path.join(out_folder,"Results.pkl"),"wb") as file:
        pickle.dump(results_dict,file)
    
    seps = gaussian_filter1d(seps, sigma=1.25)

    seps_NF = gaussian_filter1d(seps_NF,sigma=1.25)

    fig = plt.figure(figsize=(12,6))

    plt.errorbar(thetas, seps, yerr=sep_err, color='black', lw=2, 
                label=str(model_type)+r' - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.errorbar(thetas, seps_NF, yerr=NF_err, color='blue', lw=2, 
                label=r'NF-DLL - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps_NF)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.legend(fontsize=22,ncol=2,loc="lower left")
    plt.xlabel("Polar Angle [deg.]",fontsize=25,labelpad=15)
    plt.ylabel("Separation [s.d.]",fontsize=25,labelpad=15)
    plt.ylim(0,None)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if momentum == 6.0:
        plt.ylim(0,5)
    plt.title(r"$ {0} \; GeV/c$".format(int(momentum)),fontsize=28)
    plt.savefig(os.path.join(out_folder,"Seperation_{0}_NF_{1}GeV.pdf".format(str(model_type),int(momentum))),bbox_inches="tight")
    plt.close()

    print(" ")
    print(model_type)
    print("Average sigma: ",np.average(seps)," +- ",np.std(seps) / np.sqrt(len(seps)))
    print("NF")
    print("Average sigma: ",np.average(seps_NF)," +- ",np.std(seps_NF) / np.sqrt(len(seps_NF)))
    print(" ")

def compute_metrics(preds_flat,y_flat):
    TP = ((preds_flat == 1.) & (y_flat == 1.)).sum().item()
    FP = ((preds_flat == 1.) & (y_flat == 0.)).sum().item()
    FN = ((preds_flat == 0.) & (y_flat == 1.)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)


    return precision,recall,f1

def run_plotting_GlueX(out_folder,skewnorm=False):
    LL_Kaon = np.load(os.path.join(out_folder,"Kaon_DLL_Results.pkl"),allow_pickle=True)
    LL_Pion = np.load(os.path.join(out_folder,"Pion_DLL_Results.pkl"),allow_pickle=True)

    kin_p = LL_Pion['Kins']
    kin_k = LL_Kaon['Kins']
    dll_p = LL_Pion['z_value']
    dll_k = LL_Kaon['z_value']
    conditions = np.concatenate([kin_p,kin_k],0)
    print("NaN Checks: ",np.isnan(dll_k).sum())
    print("NaN Checks: ",np.isnan(dll_p).sum())
    dll_k = np.clip(dll_k[~np.isnan(dll_k)],-99999,99999)
    dll_p = np.clip(dll_p[~np.isnan(dll_p)],-99999,99999)
    kin_k =  kin_k[~np.isnan(dll_k)]
    kin_p = kin_p[~np.isnan(dll_p)]

    print("Pion max/min: ", dll_p.max(),dll_p.min())
    print("Kaon max/min: ",dll_k.max(),dll_k.min())

    
    bins = np.linspace(-20,20,400) 

    ### Raw DLL
    plt.hist(dll_k,bins=bins,density=True,alpha=1.0,label=r'$\mathcal{K}$',color='red',histtype='step',lw=2)
    plt.hist(dll_p,bins=bins,density=True,alpha=1.0,label=r'$\pi$',color='blue',histtype='step',lw=2)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()


    predictions = np.concatenate([dll_p,dll_k],axis=0)
    truth = np.concatenate([np.zeros_like(dll_p),np.ones_like(dll_k)],axis=0)
    efficiencies, rejections,auc = compute_efficiency_rejection(predictions, truth)
    #efficiencies_geom, rejections_geom, auc_geom = compute_efficiency_rejection_DLL(dll_geom, truth)

    swin_results = np.load("other_results/swin_results.pkl",allow_pickle=True)
    geom_results = np.load("other_results/geom_results.pkl",allow_pickle=True)
    nf_results = np.load("other_results/NF_results.pkl",allow_pickle=True)

    rj_swin = swin_results['rejections']
    eff_swin = swin_results['efficiencies']
    swin_auc = swin_results['auc']

    rj_geom = geom_results['rejections']
    eff_geom = geom_results['efficiencies']
    geom_auc = geom_results['auc']

    rj_fn = nf_results['rejections']
    eff_fn = nf_results['efficiencies']
    nf_auc = nf_results['auc']

    fig = plt.figure()
    #plt.plot(rejections_geom,efficiencies_geom, color='blue', lw=2, label=r'Geometric Method. AUC = {0:.3f}'.format(auc_geom))
    plt.plot(rejections,efficiencies,color='red', lw=2, label=r'FM. AUC = {0:.3f}'.format(auc))
    plt.plot(rj_swin,eff_swin,color='magenta',lw=2,label=r'Swin. AUC = {0:.3f}'.format(swin_auc))
    plt.plot(rj_fn,eff_fn,color='blue',lw=2,label=r'NF-DLL. AUC = {0:.3f}'.format(nf_auc))
    plt.plot(rj_geom,eff_geom,color='k',lw=2,label=r'Geometric. AUC = {0:.3f}'.format(geom_auc))
    plt.plot([0, 1], [1, 0], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'kaon efficiency',fontsize=25)
    plt.ylabel(r'pion rejection',fontsize=25) 
    plt.legend(loc="lower left",fontsize=14)
    plt.ylim(0,1)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    out_path = os.path.join(out_folder,"ROC.pdf")
    plt.savefig(out_path,bbox_inches='tight')
    plt.close(fig)

    # AUC function of momentum
    mom_ranges = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5]
    centers = [mr+0.25 for mr in mom_ranges[:-1]]
    aucs = []
    aucs_upper = []
    aucs_lower = []
    aucs_geom = []
    aucs_geom_upper = []
    aucs_geom_lower = []
    lengths = []
    n_kaons = []
    n_pions = []
    seps = []
    sep_err = []

    swin_results = np.load("other_results/auc_func_p_swin.pkl",allow_pickle=True)
    geom_results = np.load("other_results/auc_func_p_geom.pkl",allow_pickle=True)
    nf_results = np.load("other_results/NF_auc_func_p.pkl",allow_pickle=True)

    swin_aucs = swin_results['aucs'][:len(centers)]
    swin_uppers = swin_results['uppers'][:len(centers)]
    swin_lowers = swin_results['lowers'][:len(centers)]

    geom_aucs = geom_results['aucs'][:len(centers)]
    geom_uppers = geom_results['uppers'][:len(centers)]
    geom_lowers = geom_results['lowers'][:len(centers)]

    NF_aucs = nf_results['aucs'][:len(centers)]
    NF_uppers = nf_results['uppers'][:len(centers)]
    NF_lowers = nf_results['lowers'][:len(centers)]

    for i in range(len(mom_ranges) - 1):
        mom_low = mom_ranges[i]
        mom_high = mom_ranges[i+1]
        idx = np.where((conditions[:,0] > mom_low) & (conditions[:,0] < mom_high))[0]
        p = predictions[idx]
        #p_geom = dll_geom[idx]
        t = truth[idx]
        print("Momentum Range: ",mom_low,"-",mom_high)
        print("# Kaons: ",len(t[t==1]))
        n_kaons.append(len(t[t==1]))
        n_pions.append(len(t[t==0]))
        print("# Pions: ",len(t[t==0]))
        lengths.append(len(p))
        eff,rej,_ = compute_efficiency_rejection(p,t)#roc_curve(t,p)
        #eff_geom,rej_geom,_= compute_efficiency_rejection_DLL(p_geom,t)#roc_curve(t_geom,p_geom)
        AUC = []
        AUC_geom = []
        sigma_eff = np.sqrt(eff * (1.0 - eff) / len(t[t == 1]))
        sigma_rej = np.sqrt(rej * (1.0 - rej) / len(t[t == 0]))
        #sigma_eff_geom = np.sqrt(eff_geom * (1.0 - eff_geom) / len(t[t == 1]))
        #sigma_rej_geom = np.sqrt(rej_geom * (1.0 - rej_geom) / len(t[t == 0]))
        #print('FPR: ',fpr,'+-',sigma_fpr, " TPR: ",tpr,"+-",sigma_tpr)
        from scipy.integrate import trapezoid, simpson
        for _ in range(1000):
            eff_ = np.random.normal(eff,sigma_eff)
            rej_ = np.random.normal(rej,sigma_rej)
            #eff_geom_ = np.random.normal(eff_geom,sigma_eff_geom)
            #rej_geom_ = np.random.normal(rej_geom,sigma_rej_geom)

            AUC.append(trapezoid(y=np.flip(rej_),x=np.flip(eff_)))
            #AUC_geom.append(trapezoid(y=np.flip(rej_geom_),x=np.flip(eff_geom_)))


        aucs.append(np.mean(AUC))
        #aucs_geom.append(np.mean(AUC_geom))

        aucs_upper.append(np.percentile(AUC,97.5))
        aucs_lower.append(np.percentile(AUC,2.5))

        #aucs_geom_upper.append(np.percentile(AUC_geom,97.5))
        #aucs_geom_lower.append(np.percentile(AUC_geom,2.5))
        print("FM. -> Mean AUC: ",np.mean(AUC)," 95%",np.percentile(AUC,2.5),"-",np.percentile(AUC,97.5))
        #print("Geom. -> Mean AUC: ",np.mean(AUC_geom)," 95%",np.percentile(AUC_geom,2.5),"-",np.percentile(AUC_geom,97.5))


        # # Sigma separation
        # p_idx = np.where(t == 0.0)[0]
        # k_idx = np.where(t == 1.0)[0]
        # if not skewnorm:
        #     popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = perform_fit(p[k_idx],p[p_idx],bins)
        # else:
        #     popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = fit_skewnorm(p[k_idx],p[p_idx],bins)
        # seps.append(abs(sep_NF))
        # sep_err.append(se)

        # print(f"Momentum Range: ({mom_low:.2f},{mom_high:.2f}) - Sigma: {sep_NF:.2f} +- {se:.2f}")

        # if not skewnorm:
        #     if normalized:
        #         gaussian = gaussian_normalized
        #     else:
        #         gaussian = gaussian_unnormalized
        #     plt.plot(bin_centers_k_NF, gaussian(bin_centers_k_NF, *popt_k_NF),color='blue', label=r"$\mathcal{K}$")
        #     plt.plot(bin_centers_p_NF, gaussian(bin_centers_p_NF, *popt_p_NF),color='red', label=r"$\pi$")

        # else:
        #     x_pion,pdf_pion = plot_skewnorm(popt_p_NF, bin_centers_p_NF)
        #     x_kaon,pdf_kaon = plot_skewnorm(popt_k_NF, bin_centers_k_NF)
        #     plt.plot(x_kaon, pdf_kaon,color='blue', label=r"K")
        #     plt.plot(x_pion, pdf_pion,color='red', label=r"$\pi$")

        # plt.hist(p[p_idx],bins=bins,density=normalized,color='red',histtype='step',lw=3)
        # plt.hist(p[k_idx],bins=bins,density=normalized,color='blue',histtype='step',lw=3)
        # plt.legend(fontsize=18) 
        # plt.title(r"$|\vec{p}| \in $ " + f"({mom_low:.2f},{mom_high:.2f})" + r", $\sigma = $ {0:.2f}".format(sep_NF),fontsize=18)
        # plt.xlabel(r"$Ln \, L(K) - Ln \, L(\pi)$",fontsize=18)
        # plt.ylabel("entries [#]",fontsize=18)
        # plt.savefig(os.path.join(out_folder,f"Gauss_fit_momentum_({mom_low:.2f},{mom_high:.2f}).pdf"),bbox_inches="tight")
        # plt.close()

    fig = plt.figure(figsize=(10,10))
    #plt.errorbar(centers,aucs_geom,yerr=[np.array(aucs_geom) - np.array(aucs_geom_lower),np.array(aucs_geom_upper) - np.array(aucs_geom)],label=r"$AUC_{Geometric.}$",color='blue',marker='o',capsize=5)
    plt.errorbar(centers,aucs,yerr=[np.array(aucs) - np.array(aucs_lower),np.array(aucs_upper) - np.array(aucs)],label=r"$AUC_{FM.}$",color='red',marker='o',capsize=5)
    plt.errorbar(centers,swin_aucs,yerr=[np.array(swin_aucs) - np.array(swin_lowers),np.array(swin_uppers) - np.array(swin_aucs)],label=r"$AUC_{Swin.}$",color='magenta',marker='o',capsize=5)
    plt.errorbar(centers,NF_aucs,yerr=[np.array(NF_aucs) - np.array(NF_lowers),np.array(NF_uppers) - np.array(NF_aucs)],label=r"$AUC_{NF-DLL.}$",color='blue',marker='o',capsize=5)
    plt.errorbar(centers,geom_aucs,yerr=[np.array(geom_aucs) - np.array(geom_lowers),np.array(geom_uppers) - np.array(geom_aucs)],label=r"$AUC_{Geom.}$",color='k',marker='o',capsize=5)


    # plt.axhline(0.98, color='k', lw=2, linestyle='--',label=r"3\sigma \, separation")
    legend1 = plt.legend(loc='lower left', fontsize=24)
    legend1.get_frame().set_facecolor('white')  # Set legend facecolor
    legend1.get_frame().set_edgecolor('grey')  # Set legend edgecolor
    legend1.get_frame().set_alpha(1.0)  # Set legend alpha
    plt.xlabel("momentum [GeV/c]",fontsize=30,labelpad=10)
    plt.ylabel("AUC",fontsize=30,labelpad=10)
    plt.xticks(fontsize=20)  # adjust fontsize as needed
    plt.yticks(fontsize=20)  # adjust fontsize as needed
    # if np.min(aucs) < np.min(aucs_geom):
    #     min_aucs = np.min(aucs)
    # else:
    #     min_aucs = np.min(aucs_geom)
    # if np.max(aucs) > np.max(aucs_geom):
    #     max_aucs = np.max(aucs)
    # else:
    #     max_aucs = np.max(aucs_geom)

    min_aucs = np.min(geom_aucs + aucs + swin_aucs + NF_aucs)
    max_aucs = np.max(geom_aucs + aucs + swin_aucs + NF_aucs)

    plt.ylim(min_aucs - 0.05,max_aucs + 0.05)

    ax2 = plt.twinx()

    # Plot bars for pions and kaons
    ax2.bar(np.array(centers) - 0.1, n_pions, width=0.2, label='Pions', color='blue', alpha=0.15)
    ax2.bar(np.array(centers) + 0.1, n_kaons, width=0.2, label='Kaons', color='green', alpha=0.15)
    ax2.set_ylabel('Counts', fontsize=30,labelpad=10)
    ax2.tick_params(axis='y', labelsize=20)
    legend2 = ax2.legend(loc='upper right', fontsize=24)
    legend2.get_frame().set_facecolor('white')  # Set legend facecolor
    legend2.get_frame().set_edgecolor('grey')  # Set legend edgecolor
    legend2.get_frame().set_alpha(1.0)  # Set legend alpha
    out_path = os.path.join(out_folder,"AUC_func_P.pdf")
    plt.savefig(out_path,bbox_inches='tight')
    plt.close()



