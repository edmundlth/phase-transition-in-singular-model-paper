# Instructions for running Jupyter notebook
In this repository, `pipenv` is used as the package manager. To install `pipenv` on MacOS, run 
```
$ brew install pipenv
```

To install require package, 
```
$ cd /path/to/repository/root/    # make sure Pipfile.lock is in the current directory. 

$ pipenv install . 
```

To start a Jupyter server, either run `jupyter notebook` in a `pipenv shell`
```
$ pipenv shell
$ jupyter notebook
```

or run 
```
$ pipenv run jupyter notebook
```
