
[![Build Status](https://travis-ci.org/tgquintela/pythonUtils.svg?branch=master)](https://travis-ci.org/tgquintela/pythonUtils)
[![Coverage Status](https://coveralls.io/repos/github/tgquintela/pythonUtils/badge.svg?branch=master)](https://coveralls.io/github/tgquintela/pythonUtils?branch=master)
# pythonUtils
This package is a collection of different subpackages that they do not have connection between each other but the use to complement other codes in python.
They are useful to save time and reduce complexity in other projects in python. The subpackages contained are minimal utilities or codes that are in initial stages of development.
They wrap commonly used python libraries as numpy, matplotlib or pandas to add functionalities oriented to the tasks I usually do.


## Subpackages

* TUI_tools: package which helps to create a TUI for a python code.
* ProcessTools: Package which helps to create a process class in which we want to track an iteration.
* Logger: Package to create logs.
* ExploreDA: Package to easily compute statistics in different type of data and plot them.
* TesterResults: Package to group some tester utils in order to facilitate the task of testing numerical analysis results.
* numpy_tools: Package to complement numpy in some uncovered by numpy but useful tasks.
* parallel_tools: Package to group all the functions related to parallelize tasks.
* Combinatorics: Package to group functions to explore combinations of elements or generate combinations of elements for being used in different tasks.
* CollectionMeasures: Package with a collection of dummy measures used previously.
* Perturbations: Package with a collection of data perturbation utils in order to resampling existent data.
* perturbation_tests: Package to apply perturbation tests to our models.
* sklearn_tools: Package with complements to some sklearn utilities.


## Intallation

```Bash
git clone https://github.com/tgquintela/pythonUtils
.\install

```

## Testing
You need to ensure that all the package requirements are installed. pythonUtils provide a testing module which could be called by importing the module and applying test function.
If we are in the python idle:

```python
import pythonUtils
pythonUtils.test()
```
or from the shell
```shell
>> python -c "import pythonUtils; pythonUtils.test()"

```

for developers you can test from the source using nosetests of nose package.

```shell
nosetests path/to/dir
```

