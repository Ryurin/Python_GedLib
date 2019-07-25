# distutils: language = c++

"""
    Python GedLib module
    ======================
    
    This module allow to use a C++ library for edit distance between graphs (GedLib) with Python.

    
    Authors
    -------------------
 
    David Blumenthal
    Natacha Lambert

    Copyright (C) 2019 by all the authors

    Classes & Functions
    -------------------
 
"""

################################
##DEFINITIONS OF C++ FUNCTIONS##
################################


#Types imports for C++ compatibility
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.list cimport list

#Long unsigned int equivalent
cimport numpy as np
ctypedef np.npy_uint32 UINT32_t
from cpython cimport array

#Functions importation
cdef extern from "src/GedLibBind.h" :
    cdef vector[string] getEditCostStringOptions()
    cdef vector[string] getMethodStringOptions()
    cdef vector[string] getInitStringOptions()
    cdef bool isInitialized()
    cdef void restartEnv()
    cdef void loadGXLGraph(string pathFolder, string pathXML)
    cdef pair[size_t,size_t] getGraphIds()
    cdef vector[size_t] getAllGraphIds()
    cdef string getGraphClass(size_t id)
    cdef string getGraphName(size_t id)
    cdef size_t addGraph(string name, string classe)
    cdef void addNode(size_t graphId, string nodeId, map[string,string] nodeLabel)
    cdef void addEdge(size_t graphId, string tail, string head, map[string,string] edgeLabel, bool ignoreDuplicates)
    cdef void clearGraph(size_t graphId)
    cdef size_t getGraphInternalId(size_t graphId)
    cdef size_t getGraphNumNodes(size_t graphId)
    cdef size_t getGraphNumEdges(size_t graphId)
    cdef vector[string] getGraphOriginalNodeIds(size_t graphId)
    cdef vector[map[string, string]] getGraphNodeLabels(size_t graphId)
    cdef map[pair[size_t,size_t], map[string,string]] getGraphEdges(size_t graphId)
    cdef vector[vector[size_t]] getGraphAdjacenceMatrix(size_t graphId)
    cdef void setEditCost(string editCost, vector[double] editCostConstant)
    cdef void setPersonalEditCost(vector[double] editCostConstant)
    cdef void initEnv(string initOption)
    cdef void setMethod(string method, string options)
    cdef void initMethod()
    cdef double getInitime()
    cdef void runMethod(size_t g, size_t h)
    cdef double getUpperBound(size_t g, size_t h)
    cdef double getLowerBound(size_t g, size_t h)
    cdef vector[np.npy_uint64] getForwardMap(size_t g, size_t h)
    cdef vector[np.npy_uint64] getBackwardMap(size_t g, size_t h)
    cdef size_t getNodeImage(size_t g, size_t h, size_t nodeId)
    cdef size_t getNodePreImage(size_t g, size_t h, size_t nodeId)
    cdef size_t getDummyNode()
    cdef vector[pair[size_t,size_t]] getNodeMap(size_t g, size_t h)
    cdef vector[vector[int]] getAssignmentMatrix(size_t g, size_t h)
    cdef vector[vector[np.npy_uint64]] getAllMap(size_t g, size_t h)
    cdef double getRuntime(size_t g, size_t h)
    cdef bool quasimetricCosts()
    cdef vector[vector[size_t]] hungarianLSAP(vector[vector[size_t]] matrixCost);
    cdef vector[vector[double]] hungarianLSAPE(vector[vector[double]] matrixCost);


    
###########################################
##REDEFINITION OF C++ FUNCTIONS IN PYTHON##
###########################################

def PyIsInitialized() :
    """
        Checks and returns if the computation environment is initialized or not.
 
        :return: True if it's initialized, False otherwise
        :rtype: bool
        
        .. note:: This function exists for internals verifications but you can use it for your code. 
    """
    return isInitialized()

def PyGetEditCostOptions() :
    """
        Searchs the differents edit cost functions and returns the result.
 
        :return: The list of edit cost functions
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfEditCostOptions. 
        .. note:: Prefer the listOfEditCostOptions attribute of this module.
    """
    
    return getEditCostStringOptions()

def PyGetMethodOptions() :
    """
        Searchs the differents method for edit distance computation between graphs and returns the result.
 
        :return: The list of method to compute the edit distance between graphs
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfMethodOptions.
        .. note:: Prefer the listOfMethodOptions attribute of this module.
    """
    return getMethodStringOptions()

def PyGetInitOptions() :
    """
        Searchs the differents initialization parameters for the environment computation for graphs and returns the result.
 
        :return: The list of options to initialize the computation environment
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfInitOptions.
        .. note:: Prefer the listOfInitOptions attribute of this module.
    """
    return getInitStringOptions()

def PyRestartEnv() :
    """
        Restarts the environment variable. All data related to it will be delete. 
 
        .. warning:: This function deletes all graphs, computations and more so make sure you don't need anymore your environment. 
        .. note:: You can now delete and add somes graphs after initialization so you can avoid this function. 
    """
    restartEnv()

def PyLoadGXLGraph(pathFolder, pathXML) :
    """
        Loads some GXL graphes on the environment which is in a same folder, and present in the XMLfile. 
        
        :param pathFolder: The folder's path which contains GXL graphs
        :param pathXML: The XML's path which indicates which graphes you want to load
        :type pathFolder: string
        :type pathXML: string
 
        .. note:: You can call this function multiple times if you want, but not after an init call. 
    """
    loadGXLGraph(pathFolder.encode('utf-8'), pathXML.encode('utf-8'))

def PyGetGraphIds() :
    """
        Searchs the first and last IDs of the loaded graphs in the environment. 
 
        :return: The pair of the first and the last graphs Ids
        :rtype: tuple(size_t, size_t)
        
        .. note:: Prefer this function if you have huges structures with lots of graphs.  
    """
    return getGraphIds()

def PyGetAllGraphIds() :
    """
        Searchs all the IDs of the loaded graphs in the environment. 
 
        :return: The list of all graphs's Ids 
        :rtype: list[size_t]
        
        .. note:: The last ID is equal to (number of graphs - 1). The order correspond to the loading order. 
    """
    return getAllGraphIds()

def PyGetGraphClass(id) :
    """
        Returns the class of a graph with its ID.

        :param id: The ID of the wanted graph
        :type id: size_t
        :return: The class of the graph which correpond to the ID
        :rtype: string
        
        .. seealso:: PyGetGraphClass()
        .. note:: An empty string can be a class. 
    """
    return getGraphClass(id)

def PyGetGraphName(id) :
    """
        Returns the name of a graph with its ID. 

        :param id: The ID of the wanted graph
        :type id: size_t
        :return: The name of the graph which correpond to the ID
        :rtype: string
        
        .. seealso:: PyGetGraphClass()
        .. note:: An empty string can be a name. 
    """
    return getGraphName(id)

def PyAddGraph(name="", classe="") :
    """
        Adds a empty graph on the environment, with its name and its class. Nodes and edges will be add in a second time. 

        :param name: The name of the new graph, an empty string by default
        :param classe: The class of the new graph, an empty string by default
        :type name: string
        :type classe: string
        :return: The ID of the newly graphe
        :rtype: size_t
        
        .. seealso::PyAddNode(), PyAddEdge() , PyAddSymmetricalEdge()
        .. note:: You can call this function without parameters. You can also use this function after initialization, call PyInitEnv() after you're finished your modifications. 
    """
    return addGraph(name.encode('utf-8'),classe.encode('utf-8'))

def PyAddNode(graphID, nodeID, nodeLabel):
    """
        Adds a node on a graph selected by its ID. A ID and a label for the node is required. 

        :param graphID: The ID of the wanted graph
        :param nodeID: The ID of the new node
        :param nodeLabel: The label of the new node
        :type graphID: size_t
        :type nodeID: string
        :type nodeLabel: dict{string : string}
        
        .. seealso::PyAddGraph(), PyAddEdge(), PyAddSymmetricalEdge()
        .. note:: You can also use this function after initialization, but only on a newly added graph. Call PyInitEnv() after you're finished your modifications. 
    """
    addNode(graphID, nodeID.encode('utf-8'), encodeYourMap(nodeLabel))

def PyAddEdge(graphID, tail, head, edgeLabel, ignoreDuplicates = True) :
    """
        Adds an edge on a graph selected by its ID. 

        :param graphID: The ID of the wanted graph
        :param tail: The ID of the tail node for the new edge
        :param head: The ID of the head node for the new edge
        :param edgeLabel: The label of the new edge
        :param ignoreDuplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
        :type graphID: size_t
        :type tail: string
        :type head: string
        :type edgeLabel: dict{string : string}
        :type ignoreDuplicates: bool
        
        .. seealso::PyAddGraph(), PyAddNode(), PyAddSymmetricalEdge()
        .. note:: You can also use this function after initialization, but only on a newly added graph. Call PyInitEnv() after you're finished your modifications. 
    """
    addEdge(graphID, tail.encode('utf-8'), head.encode('utf-8'), encodeYourMap(edgeLabel), ignoreDuplicates)

def PyAddSymmetricalEdge(graphID, tail, head, edgeLabel) :
    """
        Adds a symmetrical edge on a graph selected by its ID. 

        :param graphID: The ID of the wanted graph
        :param tail: The ID of the tail node for the new edge
        :param head: The ID of the head node for the new edge
        :param edgeLabel: The label of the new edge
        :type graphID: size_t
        :type tail: string
        :type head: string
        :type edgeLabel: dict{string : string}
        
        .. seealso::PyAddGraph(), PyAddNode(), PyAddEdge()
        .. note:: You can also use this function after initialization, but only on a newly added graph. Call PyInitEnv() after you're finished your modifications. 
    """
    tailB = tail.encode('utf-8')
    headB = head.encode('utf-8')
    edgeLabelB = encodeYourMap(edgeLabel)
    addEdge(graphID, tailB, headB, edgeLabelB, True)
    addEdge(graphID, headB, tailB, edgeLabelB, True)

def PyClearGraph(graphID) :
    """
        Deletes a graph, selected by its ID, to the environment.

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        
        .. note:: Call PyInit() after you're finished your modifications. 
    """
    clearGraph(graphID)

def PyGetGraphInternalId(graphID) :
    """
        Searchs and returns the internal Id of a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The internal ID of the selected graph
        :rtype: size_t
        
        .. seealso::PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphInternalId(graphID)

def PyGetGraphNumNodes(graphID) :
    """
        Searchs and returns the number of nodes on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The number of nodes on the selected graph
        :rtype: size_t
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphNumNodes(graphID)

def PyGetGraphNumEdges(graphID) :
    """
        Searchs and returns the number of edges on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The number of edges on the selected graph
        :rtype: size_t
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphNumEdges(graphID)

def PyGetOriginalNodeIds(graphID) :
    """
        Searchs and returns all th Ids of nodes on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of IDs's nodes on the selected graph
        :rtype: list[string]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetGraphNodeLabels(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphOriginalNodeIds(graphID)

def PyGetGraphNodeLabels(graphID) :
    """
        Searchs and returns all the labels of nodes on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of labels's nodes on the selected graph
        :rtype: list[dict{string : string}]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphNodeLabels(graphID)

def PyGetGraphEdges(graphID) :
    """
        Searchs and returns all the edges on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of edges on the selected graph
        :rtype: dict{tuple(size_t,size_t) : dict{string : string}}
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphEdges(graphID)

def PyGetGraphAdjacenceMatrix(graphID) :
    """
        Searchs and returns the adjacence list of a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The adjacence list of the selected graph
        :rtype: list[list[size_t]]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphEdges()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphAdjacenceMatrix(graphID)

def PySetEditCost(editCost, editCostConstant = []) :
    """
        Sets an edit cost function to the environment, if its exists. 

        :param editCost: The name of the edit cost function
        :type editCost: string
        :param editCostConstant: The parameters you will add to the editCost, empty by default
        :type editCostConstant: list
        
        .. seealso::listOfEditCostOptions
        .. note:: Try to make sure the edit cost function exists with listOfEditCostOptions, raise an error otherwise. 
    """
    editCostB = editCost.encode('utf-8')
    if editCostB in listOfEditCostOptions : 
        setEditCost(editCostB, editCostConstant)
    else :
        raise EditCostError("This edit cost function doesn't exist, please see listOfEditCostOptions for selecting a edit cost function")

def PySetPersonalEditCost(editCostConstant = []) :
    """
        Sets an personal edit cost function to the environment.

        :param editCostConstant: The parameters you will add to the editCost, empty by default
        :type editCostConstant: list
        
        .. note::You have to modify the C++ function to use it. Please see the documentation to add your Edit Cost function. 
    """
    setPersonalEditCost(editCostConstant)

def PyInitEnv(initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    """
        Initializes the environment with the chosen edit cost function and graphs.

        :param initOption: The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
        :type initOption: string
        
        .. seealso:: listOfInitOptions
        .. warning:: No modification were allowed after initialization. Try to make sure your choices is correct. You can though clear or add a graph, but recall PyInitEnv() after that. 
        .. note:: Try to make sure the option exists with listOfInitOptions or choose no options, raise an error otherwise.
    """
    initB = initOption.encode('utf-8')
    if initB in listOfInitOptions : 
        initEnv(initB)
    else :
        raise InitError("This init option doesn't exist, please see listOfInitOptions for selecting an option. You can choose any options.")

def PySetMethod(method, options="") :
    """
        Sets a computation method to the environment, if its exists. 

        :param method: The name of the computation method
        :param options: The options of the method (like bash options), an empty string by default
        :type method: string
        :type options: string
        
        .. seealso:: PyInitMethod(), listOfMethodOptions
        .. note:: Try to make sure the edit cost function exists with listOfMethodOptions, raise an error otherwise. Call PyInitMethod() after your set. 
    """
    methodB = method.encode('utf-8')
    if methodB in listOfMethodOptions :
        setMethod(methodB, options.encode('utf-8'))
    else :
        raise MethodError("This method doesn't exist, please see listOfMethodOptions for selecting a method")

def PyInitMethod() :
    """
        Inits the environment with the set method.

        .. seealso:: PySetMethod(), listOfMethodOptions
        .. note:: Call this function after set the method. You can't launch computation or change the method after that. 
    """
    initMethod()

def PyGetInitime() :
    """
        Returns the initialization time.

        :return: The initialization time
        :rtype: double
    """
    return getInitime()

def PyRunMethod(g, h) :
    """
        Computes the edit distance between two graphs g and h, with the edit cost function and method computation selected.  

        :param g: The Id of the first graph to compare
        :param h: The Id of the second graph to compare
        :type g: size_t
        :type h: size_t
        
        .. seealso:: PyGetUpperBound(), PyGetLowerBound(),  PyGetForwardMap(), PyGetBackwardMap(), PyGetRuntime(), PyQuasimetricCost()
        .. note:: This function only compute the distance between two graphs, without returning a result. Use the differents function to see the result between the two graphs.  
    """
    runMethod(g,h)

def PyGetUpperBound(g,h) :
    """
        Returns the upper bound of the edit distance cost between two graphs g and h. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The upper bound of the edit distance cost
        :rtype: double
        
        .. seealso:: PyRunMethod(), PyGetLowerBound(), PyGetForwardMap(), PyGetBackwardMap(), PyGetRuntime(), PyQuasimetricCost()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: The upper bound is equivalent to the result of the pessimist edit distance cost. Methods are heuristics so the library can't compute the real perfect result because it's NP-Hard problem.
    """
    return getUpperBound(g,h)

def PyGetLowerBound(g,h) :
    """
         Returns the lower bound of the edit distance cost between two graphs g and h. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The lower bound of the edit distance cost
        :rtype: double
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetForwardMap(), PyGetBackwardMap(), PyGetRuntime(), PyQuasimetricCost()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: This function can be ignored, because lower bound doesn't have a crucial utility.    
    """
    return getLowerBound(g,h)

def PyGetForwardMap(g,h) :
    """
        Returns the forward map (or the half of the adjacence matrix) between nodes of the two indicated graphs. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The forward map to the adjacence matrix between nodes of the two graphs
        :rtype: list[npy_uint32]
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetLowerBound(), PyGetBackwardMap(), PyGetRuntime(), PyQuasimetricCost(), PyGetNodeMap(), PyGetAssignmentMatrix()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: I don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work ! 
    """
    return getForwardMap(g,h)

def PyGetBackwardMap(g,h) :
    """
        Returns the backward map (or the half of the adjacence matrix) between nodes of the two indicated graphs. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The backward map to the adjacence matrix between nodes of the two graphs
        :rtype: list[npy_uint32]
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetLowerBound(), PyGetForwardMap(), PyGetRuntime(), PyQuasimetricCost(), PyGetNodeMap(), PyGetAssignmentMatrix()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: I don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work ! 
    """
    return getBackwardMap(g,h)

def PyGetNodeImage(g,h,nodeID) :
    """
        Returns the node's image in the adjacence matrix, if it exists.   

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :param nodeID: The ID of the node which you want to see the image
        :type g: size_t
        :type h: size_t
        :type nodeID size_t
        :return: The ID of the image node
        :rtype: size_t
        
        .. seealso:: PyRunMethod(), PyGetForwardMap(), PyGetBackwardMap(), PyGetNodePreImage(), PyGetNodeMap(), PyGetAssignmentMatrix()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: Use BackwardMap's Node to find its images ! You can also use PyGetForwardMap() and PyGetBackwardMap().     

    """
    return getNodeImage(g, h, nodeID)

def PyGetNodePreImage(g,h,nodeID) :
    """
        Returns the node's preimage in the adjacence matrix, if it exists.   

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :param nodeID: The ID of the node which you want to see the preimage
        :type g: size_t
        :type h: size_t
        :type nodeID size_t
        :return: The ID of the preimage node
        :rtype: size_t
        
        .. seealso:: PyRunMethod(), PyGetForwardMap(), PyGetBackwardMap(), PyGetNodeImage(), PyGetNodeMap(), PyGetAssignmentMatrix()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: Use ForwardMap's Node to find its images ! You can also use PyGetForwardMap() and PyGetBackwardMap().     

    """
    return getNodePreImage(g, h, nodeID)

def PyGetDummyNode() :
    """
        Returns the ID of a dummy node.

        :return: The ID of the dummy node (18446744073709551614 for my computer, the hugest number possible)
        :rtype: size_t
        
        .. note:: A dummy node is used when a node isn't associated to an other node.      
    """
    return getDummyNode()

def PyGetNodeMap(g,h) : 
    """
        Returns the Node Map, like C++ NodeMap.   

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The Node Map between the two selected graph. 
        :rtype: list[tuple(size_t, size_t)]
        
        .. seealso:: PyRunMethod(), PyGetForwardMap(), PyGetBackwardMap(), PyGetNodeImage(), PyGetNodePreImage(), PyGetAssignmentMatrix()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: This function creates datas so use it if necessary, however you can understand how assignement works with this example.     
    """
    return getNodeMap(g, h)

def PyGetAssignmentMatrix(g,h) :
    """
        Returns the Assignment Matrix between two selected graphs g and h.   

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The Assignment Matrix between the two selected graph. 
        :rtype: list[list[int]]
        
        .. seealso:: PyRunMethod(), PyGetForwardMap(), PyGetBackwardMap(), PyGetNodeImage(), PyGetNodePreImage(), PyGetNodeMap()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: This function creates datas so use it if necessary.     
    """
    return getAssignmentMatrix(g, h)
        

def PyGetAllMap(g,h) :
    """
         Returns a vector which contains the forward and the backward maps between nodes of the two indicated graphs. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The forward and backward maps to the adjacence matrix between nodes of the two graphs
        :rtype: list[list[npy_uint32]]
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetLowerBound(),  PyGetForwardMap(), PyGetBackwardMap(), PyGetRuntime(), PyQuasimetricCost()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: This function duplicates data so please don't use it. I also don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work !  
    """
    return getAllMap(g,h)

def PyGetRuntime(g,h) :
    """
        Returns the runtime to compute the edit distance cost between two graphs g and h  

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: The runtime of the computation of edit distance cost between the two selected graphs
        :rtype: double
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetLowerBound(),  PyGetForwardMap(), PyGetBackwardMap(), PyQuasimetricCost()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
        .. note:: Python is a bit longer than C++ due to the functions's encapsulate.    
    """
    return getRuntime(g,h)

def PyQuasimetricCost() :
    """
        Checks and returns if the edit costs are quasimetric. 

        :param g: The Id of the first compared graph 
        :param h: The Id of the second compared graph
        :type g: size_t
        :type h: size_t
        :return: True if it's verified, False otherwise
        :rtype: bool
        
        .. seealso:: PyRunMethod(), PyGetUpperBound(), PyGetLowerBound(),  PyGetForwardMap(), PyGetBackwardMap(), PyGetRuntime()
        .. warning:: PyRunMethod() between the same two graph must be called before this function. 
    """
    return quasimetricCosts()

def PyHungarianLSAP(matrixCost) :
    """
        Applies the hungarian algorithm (LSAP) on a matrix Cost. 

        :param matrixCost: The matrix Cost  
        :type matrixCost: vector[vector[size_t]]
        :return: The values of rho, varrho, u and v, in this order
        :rtype: vector[vector[size_t]]
        
        .. seealso::PyHungarianLSAPE(matrixCost) 
    """
    return hungarianLSAP(matrixCost)

def PyHungarianLSAPE(matrixCost) :
    """
        Applies the hungarian algorithm (LSAPE) on a matrix Cost. 

        :param matrixCost: The matrix Cost 
        :type matrixCost: vector[vector[double]]
        :return: The values of rho, varrho, u and v, in this order
        :rtype: vector[vector[double]]
        
        .. seealso::PyHungarianLSAP(matrixCost) 
    """
    return hungarianLSAPE(matrixCost)

#####################################################################
##LISTS OF EDIT COST FUNCTIONS, METHOD COMPUTATION AND INIT OPTIONS##
#####################################################################

listOfEditCostOptions = PyGetEditCostOptions()
listOfMethodOptions = PyGetMethodOptions()
listOfInitOptions = PyGetInitOptions()


#####################
##ERRORS MANAGEMENT##
#####################

class Error(Exception):
    """
        Class for error's management. This one is general. 
    """
    pass

class EditCostError(Error) :
    """
        Class for Edit Cost Error. Raise an error if an edit cost function doesn't exist in the library (not in listOfEditCostOptions).

        :attribute message: The message to print when an error is detected.
        :type message: string
    """
    def __init__(self, message):
        """
            Inits the error with its message. 

            :param message: The message to print when the error is detected
            :type message: string
        """
        self.message = message
    
class MethodError(Error) :
    """
        Class for Method Error. Raise an error if a computation method doesn't exist in the library (not in listOfMethodOptions).

        :attribute message: The message to print when an error is detected.
        :type message: string
    """
    def __init__(self, message):
        """
            Inits the error with its message. 

            :param message: The message to print when the error is detected
            :type message: string
        """
        self.message = message

class InitError(Error) :
    """
        Class for Init Error. Raise an error if an init option doesn't exist in the library (not in listOfInitOptions).

        :attribute message: The message to print when an error is detected.
        :type message: string
    """
    def __init__(self, message):
        """
            Inits the error with its message. 

            :param message: The message to print when the error is detected
            :type message: string
        """
        self.message = message


#########################################
##PYTHON FUNCTIONS FOR SOME COMPUTATION##
#########################################

def encodeYourMap(map) :
    """
        Encodes a string dictionnary to utf-8 for C++ functions

        :param map: The map to encode
        :type map: dict{string : string}
        :return: The encoded map
        :rtype: dict{'b'string : 'b'string}

        .. note:: This function is used for type connection.  
        
    """
    res = {}
    for key, value in map.items():
        res[key.encode('utf-8')] = value.encode('utf-8')
    return res

def addRandomGraph(name, classe, listOfNodes, listOfEdges, ignoreDuplicates=True) :
    """
        Add a Graph (not GXL) on the environment. Be careful to respect the same format as GXL graphs for labelling nodes and edges. 

        :param name: The name of the graph to add, can be an empty string
        :param classe: The classe of the graph to add, can be an empty string
        :param listOfNodes: The list of nodes to add
        :param listOfEdges: The list of edges to add
        :param ignoreDuplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
        :type name: string
        :type classe: string
        :type listOfNodes: list[tuple(size_t, dict{string : string})]
        :type listOfEdges: list[tuple(tuple(size_t,size_t), dict{string : string})]
        :type ignoreDuplicates: bool
        :return: The ID of the newly added graphe
        :rtype: size_t

        .. note:: The graph must respect the GXL structure. Please see how a GXL graph is construct.  
        
    """
    id = PyAddGraph(name, classe)
    for node in listOfNodes :
        PyAddNode(id, node[0], node[1])
    for edge in listOfEdges :
        PyAddEdge(id, edge[0], edge[1], edge[2], ignoreDuplicates)
    return id

def addNxGraph(g, classe, ignoreDuplicates=True) :
    """
        Add a Graph (made by networkx) on the environment. Be careful to respect the same format as GXL graphs for labelling nodes and edges. 

        :param g: The graph to add (networkx graph)
        :param ignoreDuplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
        :type g: networkx.graph
        :type ignoreDuplicates: bool
        :return: The ID of the newly added graphe
        :rtype: size_t

        .. note:: The NX graph must respect the GXL structure. Please see how a GXL graph is construct.  
        
    """
    id = PyAddGraph(g.name, classe)
    for node in g.nodes :
        PyAddNode(id, str(node), g.node[node])
    for edge in g.edges :
        PyAddEdge(id, str(edge[0]), str(edge[1]), g.get_edge_data(edge[0],edge[1]), ignoreDuplicates)
    return id


def computeGedOnTwoGraphs(g1,g2, editCost, method, options, initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    """
        Computes the edit distance between two NX graphs. 
        
        :param g1: The first graph to add and compute
        :param g2: The second graph to add and compute
        :param editCost: The name of the edit cost function
        :param method: The name of the computation method
        :param options: The options of the method (like bash options), an empty string by default
        :param initOption:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
        :type g1: networksx.graph
        :type g2: networksx.graph
        :type editCost: string
        :type method: string
        :type options: string
        :type initOption: string
        :return: The edit distance between the two graphs and the nodeMap between them. 
        :rtype: double, list[tuple(size_t, size_t)]

        .. seealso:: listOfEditCostOptions, listOfMethodOptions, listOfInitOptions 
        .. note:: Make sure each parameter exists with your architecture and these lists : listOfEditCostOptions, listOfMethodOptions, listOfInitOptions. The structure of graphs must be similar as GXL. 
        
    """
    if PyIsInitialized() :
        PyRestartEnv()

    g = addNxGraph(g1, "")
    h = addNxGraph(g2, "")

    PySetEditCost(editCost)
    PyInitEnv(initOption)
    
    PySetMethod(method, options)
    PyInitMethod()

    resDistance = 0
    resMapping = []
    PyRunMethod(g,h)
    resDistance = PyGetUpperBound(g,h)
    resMapping = PyGetNodeMap(g,h)

    return resDistance, resMapping

def computeEditDistanceOnNxGraphs(dataset, classes, editCost, method, options, initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    """

        Computes all the edit distance between each NX graphs on the dataset. 
        
        :param dataset: The list of graphs to add and compute
        :param classes: The classe of all the graph, can be an empty string
        :param editCost: The name of the edit cost function
        :param method: The name of the computation method
        :param options: The options of the method (like bash options), an empty string by default
        :param initOption:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
        :type dataset: list[networksx.graph]
        :type classes: string
        :type editCost: string
        :type method: string
        :type options: string
        :type initOption: string
        :return: Two matrix, the first with edit distances between graphs and the second the nodeMap between graphs. The result between g and h is one the [g][h] coordinates.
        :rtype: list[list[double]], list[list[list[tuple(size_t, size_t)]]]

        .. seealso:: listOfEditCostOptions, listOfMethodOptions, listOfInitOptions 
        .. note:: Make sure each parameter exists with your architecture and these lists : listOfEditCostOptions, listOfMethodOptions, listOfInitOptions. The structure of graphs must be similar as GXL. 
        
    """
    if PyIsInitialized() :
        PyRestartEnv()

    print("Loading graphs in progress...")
    for graph in dataset :
        addNxGraph(graph, classes)
    listID = PyGetGraphIds()
    print("Graphs loaded ! ")
    print("Number of graphs = " + str(listID[1]))

    PySetEditCost(editCost)
    print("Initialization in progress...")
    PyInitEnv(initOption)
    print("Initialization terminated !")
    
    PySetMethod(method, options)
    PyInitMethod()

    resDistance = [[]]
    resMapping = [[]]
    for g in range(listID[0], listID[1]) :
        print("Computation between graph " + str(g) + " with all the others including himself.")
        for h in range(listID[0], listID[1]) :
            #print("Computation between graph " + str(g) + " and graph " + str(h))
            PyRunMethod(g,h)
            resDistance[g][h] = PyGetUpperBound(g,h)
            resMapping[g][h] = PyGetNodeMap(g,h)

    print("Finish ! The return contains edit distances and NodeMap but you can check the result with graphs'ID until you restart the environment")
    return resDistance, resMapping
    
    
    
def computeEditDistanceOnGXlGraphs(pathFolder, pathXML, editCost, method, options="", initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    """
        Computes all the edit distance between each GXL graphs on the folder and the XMl file. 
        
        :param pathFolder: The folder's path which contains GXL graphs
        :param pathXML: The XML's path which indicates which graphes you want to load
        :param editCost: The name of the edit cost function
        :param method: The name of the computation method
        :param options: The options of the method (like bash options), an empty string by default
        :param initOption:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
        :type pathFolder: string
        :type pathXML: string
        :type editCost: string
        :type method: string
        :type options: string
        :type initOption: string
        :return: The list of the first and last-1 ID of graphs
        :rtype: tuple(size_t, size_t)

        .. seealso:: listOfEditCostOptions, listOfMethodOptions, listOfInitOptions 
        .. note:: Make sure each parameter exists with your architecture and these lists : listOfEditCostOptions, listOfMethodOptions, listOfInitOptions. 
        
    """

    if PyIsInitialized() :
        PyRestartEnv()

    print("Loading graphs in progress...")
    PyLoadGXLGraph(pathFolder, pathXML)
    listID = PyGetGraphIds()
    print("Graphs loaded ! ")
    print("Number of graphs = " + str(listID[1]))

    PySetEditCost(editCost)
    print("Initialization in progress...")
    PyInitEnv(initOption)
    print("Initialization terminated !")
    
    PySetMethod(method, options)
    PyInitMethod()

    #res = []
    for g in range(listID[0], listID[1]) :
        print("Computation between graph " + str(g) + " with all the others including himself.")
        for h in range(listID[0], listID[1]) :
            #print("Computation between graph " + str(g) + " and graph " + str(h))
            PyRunMethod(g,h)
            #res.append((PyGetUpperBound(g,h), PyGetNodeMap(g,h), PyGetRuntime(g,h)))
            
    #return res

    print ("Finish ! You can check the result with each ID of graphs ! There are in the return")
    print ("Please don't restart the environment or recall this function, you will lose your results !")
    return listID


    
