Examples
==============

Before using each example, please make sure to put these lines on the beginnig of your code : 

.. code-block:: python 

  import librariesImport
  import PythonGedLib

Use your path to access it, without changing the library architecture. After that, you are ready to use the library. 

When you want to make new computation, please use this function : 

.. code-block:: python 

  PythonGedLib.PyRestartEnv()

All the graphs and results will be delete so make sure you don't need it. 

Classique case with GXL graphs
------------------------------------
.. code-block:: python 

  PythonGedLib.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
  listID = PythonGedLib.PyGetAllGraphIds()
  PythonGedLib.PySetEditCost("CHEM_1")

  PythonGedLib.PyInitEnv()

  PythonGedLib.PySetMethod("IPFP", "")
  PythonGedLib.PyInitMethod()

  g = listID[0]
  h = listID[1]

  PythonGedLib.PyRunMethod(g,h)

  print("Node Map : ", PythonGedLib.PyGetNodeMap(g,h))
  print ("Upper Bound = " + str(PythonGedLib.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(PythonGedLib.PyGetLowerBound(g,h)) + ", Runtime = " + str(PythonGedLib.PyGetRuntime(g,h)))


You can also use this function :

.. code-block:: python 

  computeEditDistanceOnGXlGraphs(pathFolder, pathXML, editCost, method, options="", initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") 
    
This function compute all edit distance between all graphs, even itself. You can see the result with some functions and graphs IDs. Please see the documentation of the function for more informations. 

Classique case with NX graphs
------------------------------------
.. code-block:: python 

  for graph in dataset :
    PythonGedLib.addNxGraph(graph, classes)
  listID = PythonGedLib.PyGetGraphIds()
  PythonGedLib.PySetEditCost("CHEM_1")

  PythonGedLib.PyInitEnv()    

  PythonGedLib.PySetMethod("IPFP", "")
  PythonGedLib.PyInitMethod()

  g = listID[0]
  h = listID[1]

  PythonGedLib.PyRunMethod(g,h)

  print("Node Map : ", PythonGedLib.PyGetNodeMap(g,h))
  print ("Upper Bound = " + str(PythonGedLib.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(PythonGedLib.PyGetLowerBound(g,h)) + ", Runtime = " + str(PythonGedLib.PyGetRuntime(g,h)))

You can also use this function :

.. code-block:: python 

  computeEditDistanceOnNxGraphs(dataset, classes, editCost, method, options, initOption = "EAGER_WITHOUT_SHUFFLED_COPIES")
    
This function compute all edit distance between all graphs, even itself. You can see the result in the return and with some functions and graphs IDs. Please see the documentation of the function for more informations. 

Or this function : 

.. code-block:: python 

  ccomputeGedOnTwoGraphs(g1,g2, editCost, method, options, initOption = "EAGER_WITHOUT_SHUFFLED_COPIES")

This function allow to compute the edit distance just for two graphs. Please see the documentation of the function for more informations. 

Add a graph from scratch
------------------------------------
.. code-block:: python 

  currentID = PythonGedLib.PyAddGraph();
  PythonGedLib.PyAddNode(currentID, "_1", {"chem" : "C"})
  PythonGedLib.PyAddNode(currentID, "_2", {"chem" : "O"})
  PythonGedLib.PyAddEdge(currentID,"_1", "_2",  {"valence": "1"} )

Please make sure as the type are the same (string for Ids and a dictionnary for labels). If you want a symmetrical graph, you can use this function to ensure the symmetry : 

.. code-block:: python 

  PyAddSymmetricalEdge(graphID, tail, head, edgeLabel) 

If you have a Nx structure, you can use directly this function : 

.. code-block:: python 

  addNxGraph(g, classe, ignoreDuplicates=True)

Even if you have another structure, you can use this function : 

.. code-block:: python
 
  addRandomGraph(name, classe, listOfNodes, listOfEdges, ignoreDuplicates=True)

Please read the documentation before using and respect the types.

Median computation
------------------------------------

Coming soon ... 
Please ask Benoît Gauzere for this example ! 


Hungarian algorithm
------------------------------------


LSAPE
~~~~~~

.. code-block:: python

  result = PythonGedLib.PyHungarianLSAPE(matrixCost) 
  print("Rho = ", result[0], " Varrho = ", result[1], " u = ", result[2], " v = ", result[3])


LSAP
~~~~~~

.. code-block:: python

  result = PythonGedLib.PyHungarianLSAP(matrixCost) 
  print("Rho = ", result[0], " u = ", result[1], " v = ", result[2], " Varrho = ", result[3])



