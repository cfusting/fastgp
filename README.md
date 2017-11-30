# Fast Genetic Programming
fastgp is a numpy implementation of [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) built on top of [deap](https://github.com/DEAP/deap). It is the core library for [fastsr](https://github.com/cfusting/fast-symbolic-regression), a symbolic regression package for Python.
It's primary contribution is an implementation of AFPO<a href="#lc-1">\[1\]</a> which is compatible with any deap toolbox.

fastgp was designed and developed by the [Morphology, Evolution & Cognition Laboratory](http://www.meclab.org/) at the University of Vermont. It extends research code which can be found [here](https://github.com/mszubert/gecco_2016).

Installing
----------
fastgp is compatible with Python 2.7+.
```bash
pip install fastgp
```

Example Usage
-------------
fastgp is a core library and as such there are no examples in this repository.
Check out [fastsr](https://github.com/cfusting/fast-symbolic-regression) for an example of fastgp's use in Symbolic Regression.

Literature Cited
----------------
1. Michael Schmidt and Hod Lipson. 2011. Age-fitness pareto optimization. In Genetic Programming Theory and Practice VIII. Springer, 129–146.<a name="lc-2"></a>
