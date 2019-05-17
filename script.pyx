# distutils: language = c++

"""
    This module allow to use a C++ library for edit distance between graphs (GedLib) with Python.

    
    Authors
    -------------------
 
    David Blumenthal,
    Natacha Lambert
 
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

#Functions importation
cdef extern from "src/essai.h" :
    cdef vector[string] getEditCostStringOptions()
    cdef vector[string] getMethodStringOptions()
    cdef vector[string] getInitStringOptions()
    cdef bool isInitialized()
    cdef int appelle()
    cdef void restartEnv()
    cdef void loadGXLGraph(string pathFolder, string pathXML)
    cdef vector[size_t] getGraphIds()
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
    cdef vector[pair[pair[size_t,size_t], map[string, string]]] getGraphEdges(size_t graphId)
    cdef vector[list[pair[size_t, map[string, string]]]] getGraphAdjacenceList(size_t graphId)
    cdef void setEditCost(string editCost)
    cdef void initEnv(string initOption)
    cdef void setMethod(string method, string options)
    cdef void initMethod()
    cdef double getInitime()
    cdef void runMethod(size_t g, size_t h)
    cdef double getUpperBound(size_t g, size_t h)
    cdef double getLowerBound(size_t g, size_t h)
    cdef vector[np.npy_uint64] getForwardMap(size_t g, size_t h)
    cdef vector[np.npy_uint64] getBackwardMap(size_t g, size_t h)
    cdef vector[vector[np.npy_uint64]] getAllMap(size_t g, size_t h)
    cdef double getRuntime(size_t g, size_t h)
    cdef bool quasimetricCosts()

    
###########################################
##REDEFINITION OF C++ FUNCTIONS IN PYTHON##
###########################################
    
def appel() :
    """
        Call an example only in C++. Nothing usefull, that's why you must ignore this function. 
    """
    appelle()

def PyIsInitialized() :
    """
        Check and return if the computation environment is initialized or not.
 
        :return: True if it's initialized, False otherwise
        :rtype: bool
        
        .. note:: This function exists for internals verifications but you can use it for your code. 
    """
    return isInitialized()

def PyGetEditCostOptions() :
    """
        Search the differents edit cost functions and returns the result.
 
        :return: The list of edit cost functions
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfEditCostOptions. 
        .. note:: Prefer the listOfEditCostOptions attribute of this module.
    """
    
    return getEditCostStringOptions()

def PyGetMethodOptions() :
    """
        Search the differents method for edit distance computation between graphs and returns the result.
 
        :return: The list of method to compute the edit distance between graphs
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfMethodOptions.
        .. note:: Prefer the listOfMethodOptions attribute of this module.
    """
    return getMethodStringOptions()

def PyGetInitOptions() :
    """
        Search the differents initialization parameters for the environment computation for graphs and returns the result.
 
        :return: The list of options to initialize the computation environment
        :rtype: list[string]
 
        .. warning:: This function is useless for an external use. Please use directly listOfInitOptions.
        .. note:: Prefer the listOfInitOptions attribute of this module.
    """
    return getInitStringOptions()

def PyRestartEnv() :
    """
        Restart the environment variable. All data related to it will be delete. 
 
        .. warning:: This function deletes all graphs, computations and more so make sure you don't need anymore your environment. 
        .. note:: You can now delete and add somes graphs after initialization so you can avoid this function. 
    """
    restartEnv()

def PyLoadGXLGraph(pathFolder, pathXML) :
    """
        Load some GXL graphes on the environment which is in a same folder, and present in the XMLfile. 
        
        :param pathFolder: The folder's path which contains GXL graphs
        :param pathXML: The XML's path which indicates which graphes you want to load
        :type pathFolder: string
        :type pathXML: string
 
        .. note:: You can call this function multiple times if you want, but not after an init call. 
    """
    loadGXLGraph(pathFolder.encode('utf-8'), pathXML.encode('utf-8'))

def PyGetGraphIds() :
    """
        Search all the IDs of the loaded graphs in the environment. 
 
        :return: The list of all graphs's Ids 
        :rtype: list[size_t]
        
        .. note:: The last ID is equal to (number of graphs - 1). The order correspond to the loading order. 
    """
    return getGraphIds()

def PyGetGraphClass(id) :
    """
        Return the class of a graph with its ID.

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
        Return the name of a graph with its ID. 

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
        Add a empty graph on the environment, with its name and its class. Nodes and edges will be add in a second time. 

        :param name: The name of the new graph, an empty string by default
        :param classe: The class of the new graph, an empty string by default
        :type name: string
        :type classe: string
        :return: The ID of the newly graphe
        :rtype: size_t
        
        .. seealso::PyAddNode(), PyAddEdge()
        .. note:: You can call this function without parameters. You can also use this function after initialization, call PyInitEnv() after you're finished your modifications. 
    """
    return addGraph(name,classe)

def PyAddNode(graphID, nodeID, nodeLabel):
    """
        Add a node on a graph selected by its ID. A ID and a label for the node is required. 

        :param graphID: The ID of the wanted graph
        :param nodeID: The ID of the new node
        :param nodeLabel: The label of the new node
        :type graphID: size_t
        :type nodeID: string
        :type nodeLabel: map[string,string]
        
        .. seealso::PyAddGraph(), PyAddEdge()
        .. note:: You can also use this function after initialization, but only on a newly added graph. Call PyInitEnv() after you're finished your modifications. 
    """
    addNode(graphID, nodeID, nodeLabel)

def PyAddEdge(graphID, tail, head, edgeLabel, ignoreDuplicates = True) :
    """
        Add an edge on a graph selected by its ID. 

        :param graphID: The ID of the wanted graph
        :param tail: The ID of the tail node for the new edge
        :param head: The ID of the head node for the new edge
        :param edgeLabel: The label of the new edge
        :param ignoreDuplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
        :type graphID: size_t
        :type tail: string
        :type head: string
        :type edgeLabel: map[string,string]
        :type ignoreDuplicates: bool
        
        .. seealso::PyAddGraph(), PyAddNode()
        .. note:: You can also use this function after initialization, but only on a newly added graph. Call PyInitEnv() after you're finished your modifications. 
    """
    addEdge(graphID, tail, head, edgeLabel, ignoreDuplicates)

def PyClearGraph(graphID) :
    """
        Delete a graph, selected by its ID, to the environment.

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        
        .. note:: Call PyInit() after you're finished your modifications. 
    """
    clearGraph(graphID)

def PyGetGraphInternalId(graphID) :
    """
        Search and return the internal Id of a graph, selected by its ID. 

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
        Search and return the number of nodes on a graph, selected by its ID. 

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
        Search and return the number of edges on a graph, selected by its ID. 

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
        Search and return all th Ids of nodes on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of IDs's nodes on the selected graph
        :rtype: vector[string]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetGraphNodeLabels(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphOriginalNodeIds(graphID)

def PyGetGraphNodeLabels(graphID) :
    """
        Search and return all the labels of nodes on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of labels's nodes on the selected graph
        :rtype: vector[map[string,string]]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphEdges(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphNodeLabels(graphID)

def PyGetGraphEdges(graphID) :
    """
        Search and return all th edges on a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The list of edges on the selected graph
        :rtype: vector[pair[pair[size_t,size_t], map[string, string]]]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphAdjacenceList()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphEdges(graphID)

def PyGetGraphAdjacenceList(graphID) :
    """
        Search and return the adjacence list of a graph, selected by its ID. 

        :param graphID: The ID of the wanted graph
        :type graphID: size_t
        :return: The adjacence list of the selected graph
        :rtype: vector[list[pair[size_t, map[string, string]]]]
        
        .. seealso::PyGetGraphInternalId(), PyGetGraphNumNodes(), PyGetGraphNumEdges(), PyGetOriginalNodeIds(), PyGetGraphNodeLabels(), PyGetGraphEdges()
        .. note:: These functions allow to collect all the graph's informations.
    """
    return getGraphAdjacenceList(graphID)

def PySetEditCost(editCost) :
    """
        Set an edit cost function to the environment, if its exists. 

        :param editCost: The name of the edit cost function
        :type editCost: string
        
        .. seealso::listOfEditCostOptions
        .. note:: Try to make sure the edit cost function exists with listOfEditCostOptions, raise an error otherwise. 
    """
    editCostB = editCost.encode('utf-8')
    if editCostB in listOfEditCostOptions : 
        setEditCost(editCostB)
    else :
        raise EditCostError("This edit cost function doesn't exist, please see listOfEditCostOptions for selecting a edit cost function")

def PyInitEnv(initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    """
        Initialize the environment with the chosen edit cost function and graphs.

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
        Set a computation method to the environment, if its exists. 

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
        Init the environment with the set method.

        .. seealso:: PySetMethod(), listOfMethodOptions
        .. note:: Call this function after set the method. You can't launch computation or change the method after that. 
    """
    initMethod()

def PyGetInitime() :
    return getInitime()

def PyRunMethod(g, h) :
    runMethod(g,h)

def PyGetUpperBound(g,h) :
    return getUpperBound(g,h)

def PyGetLowerBound(g,h) :
    return getLowerBound(g,h)

def PyGetForwardMap(g,h) :
    return getForwardMap(g,h)

def PyGetBackwardMap(g,h) :
    return getBackwardMap(g,h)

def PyGetAllMap(g,h) :
    return getAllMap(g,h)

def PyGetRuntime(g,h) :
    return getRuntime(g,h)

def PyQuasimetricCost() :
    return quasimetricCosts()

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
    pass

class EditCostError(Error) :
    def __init__(self, message):
        self.message = message
    
class MethodError(Error) :
    def __init__(self, message):
        self.message = message

class InitError(Error) :
    def __init__(self, message):
        self.message = message


#########################################
##PYTHON FUNCTIONS FOR SOME COMPUTATION##
#########################################

    
def computeEditDistanceOnGXlGraphs(pathFolder, pathXML, editCost, method, options, initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :

    PyRestartEnv()
    
    PyLoadGXLGraph(pathFolder, pathXML)
    listID = PyGetGraphIds()
    print("Number of graphs = " + str(len(listID)))

    PySetEditCost(editCost)
    PyInitEnv(initOption)
    
    PySetMethod(method, options)
    PyInitMethod()

    res = []
    for g in listID :
        for h in listID :
            PyRunMethod(g,h)
            res.append((PyGetUpperBound(g,h), PyGetForwardMap(g,h), PyGetBackwardMap(g,h), PyGetRuntime(g,h)))
            
    return res

    
