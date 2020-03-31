
class ML_and_stat_tools:
    """
    This class contains some ML and statistical tools to analyze Kepler/K2 data, specially lightcurve time series.
    
    Doug Souza, March 2020
    """

    def __init__(self, 
                 EMD_config={"MAX_ITERATION": 5000} 
                ):

        
        self.default_EMD_config = EMD_config # Determining EMD configuration
        

    def trends_by_EMD(self,original_time_series):
        """
        This function uses Empirical Mode Decomposition (EMD) to approximate trends, following the approach established in [1].
        For now, only the "ratio approach" is implemented (for details, see [1]).
        
        [1] "Trend Filtering via Empirical Mode Decompositions", Azadeh Moghtaderi, Patrick Flandrin, Pierre Borgnat, 
        Computational Statistics & Data Analysis, Volume 58, Pages 114–126, February 2013.
    
        Inputs: array or list
            original_time_series: univariate time series for which we want to approximate a trend component
        
        Output: array
            trend_component: approximated trend component
            detrended_time_series: detrended version of the time series
            
        Version 1.0, Doug Souza, March 2020
        """
    
        # Importing necessary modules
        import numpy as np
        from PyEMD import EEMD
    
        # Defining input parameters (hardcoded for now)
        my_config = self.default_EMD_config
        my_config["std_thr"] = 0.01*np.nanvar(original_time_series)
    
        # Computing the intrisic mode functions (IMFs) using EMD
        eemd = EEMD(**my_config) # Initializing EMD
        IMFs = eemd(original_time_series)

        # Computing energy of each individual IMF
        IMFs_energy_vector = np.mean(IMFs**2, axis = 1)

        # Finding first IMF index for which we observed an increase in energy
        IMF_index = list(np.sign(np.diff(IMFs_energy_vector))).index(1) + 1

        # Approximating trend component
        trend_component = np.sum(IMFs[IMF_index:,],axis=0)

        # Computing detrended time series
        detrended_time_series = original_time_series - trend_component

        return(trend_component, detrended_time_series)