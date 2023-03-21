---
layout: post
title: "CLIP install error"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 1.2.2.6. Transformer]
---

### Error message
``` bash
$ pip install git+https://github.com/openai/CLIP.git

Collecting git+https://github.com/openai/CLIP.git
  Cloning https://github.com/openai/CLIP.git to /tmp/pip-req-build-_hke7brn
  Running command git clone -q https://github.com/openai/CLIP.git /tmp/pip-req-build-_hke7brn
    ERROR: Command errored out with exit status 1:
     command: /usr/bin/python3 -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-req-build-_hke7brn/setup.py'"'"'; __file__='"'"'/tmp/pip-req-build-_hke7brn/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base /tmp/pip-req-build-_hke7brn/pip-egg-info
         cwd: /tmp/pip-req-build-_hke7brn/
    Complete output (54 lines):
    running egg_info
    creating /tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info
    writing /tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info/PKG-INFO
    writing dependency_links to /tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info/dependency_links.txt
    writing requirements to /tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info/requires.txt
    writing top-level names to /tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info/top_level.txt
    writing manifest file '/tmp/pip-req-build-_hke7brn/pip-egg-info/clip.egg-info/SOURCES.txt'
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-req-build-_hke7brn/setup.py", line 6, in <module>
        setup(
      File "/usr/lib/python3/dist-packages/setuptools/__init__.py", line 144, in setup
        return distutils.core.setup(**attrs)
      File "/usr/lib/python3.8/distutils/core.py", line 148, in setup
        dist.run_commands()
      File "/usr/lib/python3.8/distutils/dist.py", line 966, in run_commands
        self.run_command(cmd)
      File "/usr/lib/python3.8/distutils/dist.py", line 985, in run_command
        cmd_obj.run()
      File "/usr/lib/python3/dist-packages/setuptools/command/egg_info.py", line 297, in run
        self.find_sources()
      File "/usr/lib/python3/dist-packages/setuptools/command/egg_info.py", line 304, in find_sources
        mm.run()
      File "/usr/lib/python3/dist-packages/setuptools/command/egg_info.py", line 535, in run
        self.add_defaults()
      File "/usr/lib/python3/dist-packages/setuptools/command/egg_info.py", line 571, in add_defaults
        sdist.add_defaults(self)
      File "/usr/lib/python3.8/distutils/command/sdist.py", line 226, in add_defaults
        self._add_defaults_python()
      File "/usr/lib/python3/dist-packages/setuptools/command/sdist.py", line 135, in _add_defaults_python
        build_py = self.get_finalized_command('build_py')
      File "/usr/lib/python3.8/distutils/cmd.py", line 298, in get_finalized_command
        cmd_obj = self.distribution.get_command_obj(command, create)
      File "/usr/lib/python3.8/distutils/dist.py", line 857, in get_command_obj
        klass = self.get_command_class(command)
      File "/usr/lib/python3/dist-packages/setuptools/dist.py", line 834, in get_command_class
        self.cmdclass[command] = cmdclass = ep.load()
      File "/usr/lib/python3/dist-packages/pkg_resources/__init__.py", line 2445, in load
        return self.resolve()
      File "/usr/lib/python3/dist-packages/pkg_resources/__init__.py", line 2451, in resolve
        module = __import__(self.module_name, fromlist=['__name__'], level=0)
      File "/usr/lib/python3/dist-packages/setuptools/command/build_py.py", line 15, in <module>
        from setuptools.lib2to3_ex import Mixin2to3
      File "/usr/lib/python3/dist-packages/setuptools/lib2to3_ex.py", line 12, in <module>
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
      File "/usr/lib/python3.8/lib2to3/refactor.py", line 25, in <module>
        from .pgen2 import driver, tokenize, token
      File "/usr/lib/python3.8/lib2to3/pgen2/driver.py", line 26, in <module>
        from . import grammar, parse, token, tokenize, pgen
      File "/usr/lib/python3.8/lib2to3/pgen2/grammar.py", line 19, in <module>
        from . import token
      File "/usr/lib/python3.8/lib2to3/pgen2/token.py", line 74, in <module>
        for _name, _value in list(globals().items()):
    RuntimeError: dictionary changed size during iteration
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
```

### Solution

```
pip install --upgrade setuptools
```