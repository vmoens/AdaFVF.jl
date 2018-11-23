# AdaFVF.jl

This is a Flux compatible implementation of the AdaFVF, a HAFVF version of the Adam optimiser. 

In short, the AdaFVF will achieve some hierarchical Bayesian inference about the gradient distribution, which allows for an data-driven adaptation of the moments $\beta$1 and $\beta$2.

This is a development package, without guarantee of any kind. 

Installation
------------
This is an unregistered package. 

To install it, first clone the git repository:
```
$ git clone https://github.com/vmoens/AdaFVF.jl/
$ cd AdaFVF.jl
```

Checkout to the `dev` branch.
```
$ git checkout dev
```

To use locally, activate and instantiate:
```
$ julia
julia> ]
(v1.0) pkg> activate .
(AdaFVF) pkg> instantiate
```

Now you can add this project to your register. From any folder, run
``
julia> ]
(v1.0) pkg> /path/to/my/AdaFVF.jl
```

Usage
-----
First load the package
```
julia> using AdaFVF
```
Given a `Flux.jl` model `MyModel`, get the parameters and create an optimiser:
```
julia> P = params(MyModel)
julia> adafvf = Adafvf(P)
```

Now `adafvf` is an optimiser object associated with a method for updating the gradient. The following code update the parameters of the model:
```
julia> loss = MyModel(some_data)
julia> Flux.Tracker.back!(loss)
julia> adafvf()
```


Discosure
---------

The package is not yet formatted for Float32 real numbers: caution should be taken when used for GPU optimisation.

Publications
------------ 

If you find AdaFVF useful in your work, we kindly request that you cite

```
@InProceedings{pmlr-v80-moens18a,
  title = 	 {The Hierarchical Adaptive Forgetting Variational Filter},
  author = 	 {Moens, Vincent},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {3606--3615},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR},
  pdf = 	 {https://arxiv.org/pdf//1805.05703.pdf},
  url = 	 {http://proceedings.mlr.press/v80/moens18a.html}
}
```

