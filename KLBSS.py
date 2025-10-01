import src.KLBSS_ld as KLBSS_ld
import src.KLBSS_IC_ld as KLBSS_IC_ld
import src.KLBSS_mip as KLBSS_mip
import src.KLBSS_IC_mip as KLBSS_IC_mip
import src.CVKLBSS_ld as CVKLBSS_ld
import src.CVKLBSS_IC_ld as CVKLBSS_IC_ld
import src.CVKLBSS_mip as CVKLBSS_mip
import src.CVKLBSS_IC_mip as CVKLBSS_IC_mip
import numpy as np

def KLBSS(X,Y,
        bss=False,klbss_type='vanilla',
        s=None,ubs=None,ic='BIC',cc=None,
        betam=None,betam_ratio=None,
        cv=False,betams=None,K=5,
        mip=False,M=10,
        time_limit=600, mip_gap=1e-5, 
        hard_limit=900, hard_gap=1e-9, 
        nthread=8):
        """
        X: design matrix
        Y: response vector
        bss: whether to use BSS
        klbss_type: type of KLBSS without MIP (vanilla, simple, full)
        s: known sparsity
        ubs: sparsity upper bound
        ic: information criterion to use when sparsity is unknown (BIC, EBIC, Delta)
        cc: constant for information criterion with using Delta
        betam: betamin scalar
        betam_ratio: betamin vector is given by betam * betam_ratio, default is all 1s
        cv: whether to use cross-validation
        betams: candidate betamins when using cross-validation
        K: number of folds in cross-validation
        mip: whether to use MIP
        M: upper bound of entries in beta vector

        -- Below are the parameters for MIP -- 
        time_limit: time limit
        mip_gap: MIP gap
        hard_limit: hard limit
        hard_gap: hard gap
        nthread: number of threads
        --------------------------------------

        Returns: estimated support in sorted list
        """
        
        n, d = X.shape
        if betam_ratio is None:
            betam_ratio = np.ones(d)

        # if sparsity is known
        if s is not None:
            if bss:
                if mip:
                    return KLBSS_mip.klBSS(Y,X,s,True,None,None,None,
                                            time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
                else:
                    return KLBSS_ld.BSS(X,Y,s)
            else:
                if cv:
                    if mip:
                        return CVKLBSS_mip.CV_KLBSS(X,Y,s,betams,K,KLBSS_mip.klBSS,M,betam_ratio,
                                                time_limit,mip_gap,hard_limit,hard_gap,nthread)
                    else:
                        if klbss_type == 'vanilla':
                            return CVKLBSS_ld.CV_KLBSS(X,Y,s,betams,K,KLBSS_ld.KLBSS_vanilla,betam_ratio)
                        elif klbss_type == 'simple':
                            return CVKLBSS_ld.CV_KLBSS(X,Y,s,betams,K,KLBSS_ld.KLBSS_simple,betam_ratio)
                        elif klbss_type == 'full':
                            return CVKLBSS_ld.CV_KLBSS(X,Y,s,betams,K,KLBSS_ld.KLBSS_full,betam_ratio)
                else:
                    if mip:
                        return KLBSS_mip.klBSS(Y,X,s,False,betam*betam_ratio,betam*betam_ratio,M,
                                                time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
                    else:
                        if klbss_type == 'vanilla':
                            return KLBSS_ld.KLBSS_vanilla(X,Y,betam*betam_ratio,s)
                        elif klbss_type == 'simple':
                            return KLBSS_ld.KLBSS_simple(X,Y,betam*betam_ratio,s)
                        elif klbss_type == 'full':
                            return KLBSS_ld.KLBSS_full(X,Y,betam*betam_ratio,s)
                            
        # if sparsity is unknown
        else:
            if bss:
                if ic == 'BIC':
                    cc = np.log(n) / n
                elif ic == 'EBIC':
                    cc = np.log(d) / n

                if mip:
                    return KLBSS_IC_mip.klBSS_IC(Y,X,ubs,cc,True,None,None,None,
                                                time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
                else:
                    return KLBSS_IC_ld.BSS_IC(X,Y,ubs,cc)
            else:
                if cv:
                    if mip:
                        return CVKLBSS_IC_mip.CV_KLBSS_IC(X,Y,betams,K,KLBSS_IC_mip.klBSS_IC,ubs,ic,cc,M,betam_ratio,
                                                        time_limit,mip_gap,hard_limit,hard_gap,nthread)
                    else:
                        if klbss_type == 'vanilla':
                            return CVKLBSS_IC_ld.CV_KLBSS_IC(X,Y,betams,K,KLBSS_IC_ld.KLBSS_vanilla_IC,
                                                            ubs,ic,cc,betam_ratio)
                        elif klbss_type == 'simple':
                            return CVKLBSS_IC_ld.CV_KLBSS_IC(X,Y,betams,K,KLBSS_IC_ld.KLBSS_simple_IC,
                                                            ubs,ic,cc,betam_ratio)
                        elif klbss_type == 'full':
                            return CVKLBSS_IC_ld.CV_KLBSS_IC(X,Y,betams,K,KLBSS_IC_ld.KLBSS_full_IC,
                                                            ubs,ic,cc,betam_ratio)
                else:
                    if ic == 'BIC':
                        cc = np.log(n) / n
                    elif ic == 'EBIC':
                        cc = np.log(d) / n

                    if mip:
                        return KLBSS_IC_mip.klBSS_IC(Y,X,ubs,cc,False,betam*betam_ratio,betam*betam_ratio,M,
                                                time_limit,mip_gap,hard_limit,hard_gap,nthread)[0]
                    else:
                        if klbss_type == 'vanilla':
                            return KLBSS_IC_ld.KLBSS_vanilla_IC(X,Y,betam*betam_ratio,ubs,cc)
                        elif klbss_type == 'simple':
                            return KLBSS_IC_ld.KLBSS_simple_IC(X,Y,betam*betam_ratio,ubs,cc)
                        elif klbss_type == 'full':
                            return KLBSS_IC_ld.KLBSS_full_IC(X,Y,betam*betam_ratio,ubs,cc)