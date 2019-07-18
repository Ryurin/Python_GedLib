Python_GedLib
====================================

Please Read https://dbblumenthal.github.io/gedlib/ before using Python code.
You can also find this module documentation in documentation/build/html folder. 

Running the script
------------------

After donwloafding the entire folder, you can run test.py to ensure the library works. 

For your code, you have to make two import in your code::

  import librariesImport
  import PythonGedLib

You can call each function in the library with this import. You can't move any folder or files on the library, please make sure to keep the architecture as the same. 

This library is compiled to Python3 only. If you want to use it une Python 2, you can to recompile it with setup.py. You have to use this commad on your favorite shell::

  python setup.py build_ext --inplace

After this step, you can use the same line as Python3 for import, it will be ok. Check the documentation inside the documentation/build/html folder before using function. You can also copy the tests examples for basic use.


A problem with the library ? 
------------------

Maybe the version of GedLib or another library can be a problem. If it is, you can re-install GedLib for your computer. You can download it on this git : https://dbblumenthal.github.io/gedlib/

You have to install Gedlib with the Python installer after that. 
Just call::

  python3 install.py

Make the links like indicate on the documentation. Use the same architecture like me, but just change the .so and folders with your installation.

After that, if it's doesn't work, you can recompile the Python library. Please delete PythonGedLib.so, PythonGedLib.cpp and build folder. Then use this command on a linux shell ::

  python3 setup.py build_ext --inplace

You can make it with Python 2 but make sure you use the same version with your code and the compilation.

If you have a problem, you can contact me on : natacha.lambert@unicaen.fr


An advice if you don't code in a shell
------------------

Python library don't indicate each C++ error. If you have a restart causing by an error in your code, please use on a linux shell for having C++ errors. 
