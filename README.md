# chchanges
---
## Detect statistically meaningful changes in streams of data.

### Detect changes in the mean of normally distributed data
![mean_changepoint_probability](chchanges/demos/mean_changepoint_probability.gif)
![mean_data_stream](chchanges/demos/mean_data_stream.gif)
![mean_posterior_distribution](chchanges/demos/mean_posterior_distribution.gif)

Run the mean demo with `python mean.py` in `chchanges/demos`


### Detect changes in the variance of normally distributed data
![variance_changepoint_probability](chchanges/demos/variance_changepoint_probability.gif)
![variance_data_stream](chchanges/demos/variance_data_stream.gif)
![variance_posterior_distribution](chchanges/demos/variance_posterior_distribution.gif)

Run the variance demo with `python variance.py` in `chchanges/demos`

### References:
[Ryan P. Adams, David J.C. MacKay, "Bayesian Online Changepoint Detection" (2007)](https://arxiv.org/abs/0710.3742)
[Byrd, M Gentry et al. “Lagged Exact Bayesian Online Changepoint Detection with Parameter Estimation” (2017)](https://arxiv.org/abs/1710.03276)


### Other software implementations:
[https://github.com/dtolpin/bocd](https://github.com/dtolpin/bocd), Particularly indebted to this implementation.
[https://github.com/hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)
[https://github.com/lnghiemum/LEXO](https://github.com/lnghiemum/LEXO)

---
"turn and face the strange" - David Bowie