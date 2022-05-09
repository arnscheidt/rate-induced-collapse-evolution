This folder contains Python and Julia code to replicate results from Arnscheidt and Rothman "Rate-induced collapse in evolutionary systems".

Results from the dynamical-system model(Figs. 1-4) can be obtained by running `ds.py`. Note that the calculations for Figure 4 take a while: they are currently commented out, and saved data from `sm_critical_rate.npy` is used instead. 

The many-agent coevolutionary model is included in `ev_types.jl`. The best way to run this is by using the Julia REPL and calling specific functions; specific instructions are given in the comments. 

Finally, `graphs.py` plots outputs from the Julia model that have been saved to .csv format (using the provided functions)

Please contact Constantin Arnscheidt (cwa@mit.edu) for any questions/issues.
