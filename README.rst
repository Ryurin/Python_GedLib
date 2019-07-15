Python_GedLib
====================================

Please Read https://dbblumenthal.github.io/gedlib/ before using Python code.
You can also find this module documentation in documentation/build/html folder. 

Running the script
------------------

You have to install Gedlib with the Python installer after donwload it. 
Just call::

  python3 install.py

Make the links like indicate on the documentation. Use the same architecture like me, but just change the .so and folders with your installation.

For the use, you can copy my test.py. Please use on a linux shell for having C++ errors.

I'm not sure you have to recompile but if you have to, please delete script.so, script.cpp and build folder. Then use this command on a linux shell ::

  python3 setup.py build_ext --inplace

You can make it with Python 2 but make sure you use the same version with your code and the compilation.

For C++ tests, you can copy my essai.cpp. 

If you have a problem, you can contact me on : natacha.lambert@unicaen.fr

I hope it'll work :) ! I don't try to export all the project (because links are very annoying !) but I think, with these instructions, it'll be okay.

Good luck !
