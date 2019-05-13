# distutils: language = c++

################################
##DEFINITION DES FONCTIONS C++##
################################

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

import ctypes
ctypes.c_ulong

#import numpy as np
cimport numpy as np

ctypedef np.npy_uint32 UINT32_t

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
    cdef void setEditCost(string editCost)
    cdef void initEnv(string initOption)
    cdef void setMethod(string method, string options)
    cdef void initMethod()
    cdef double getInitime()
    cdef void runMethod(size_t g, size_t h)
    cdef double getUpperBound(size_t g, size_t h)
    cdef double getLowerBound(size_t g, size_t h)
    cdef vector[vector[np.npy_uint64]] getAllMap(size_t g, size_t h)
    cdef double getRuntime(size_t g, size_t h)
    cdef bool quasimetricCosts()

    
############################################
##REDEFINITION DES FONCTIONS C++ EN PYTHON##
############################################
    
def appel() :
    appelle()

def PyIsInitialized() :
    return isInitialized()

def PyGetEditCostOptions() :
    return getEditCostStringOptions()

def PyGetMethodOptions() :
    return getMethodStringOptions()

def PyGetInitOptions() :
    return getInitStringOptions()

def PyRestartEnv() :
    restartEnv()

def PyLoadGXLGraph(pathFolder, pathXML) :
    loadGXLGraph(pathFolder.encode('utf-8'), pathXML.encode('utf-8'))

def PyGetGraphIds() :
    return getGraphIds()

def PyGetGraphClass(id) :
    return getGraphClass(id)

def PyGetGraphName(id) :
    return getGraphName(id)

def PyAddGraph(name, classe) :
    return addGraph(name,classe)

def PySetEditCost(editCost) :
    editCostB = editCost.encode('utf-8')
    if editCostB in listOfEditCostOptions : 
        setEditCost(editCostB)
    else :
        raise EditCostError("This edit cost function doesn't exist, please see listOfEditCostOptions for selecting a edit cost function")

def PyInitEnv(initOption = "EAGER_WITHOUT_SHUFFLED_COPIES") :
    initB = initOption.encode('utf-8')
    initEnv(initB)

def PySetMethod(method, options) :
    methodB = method.encode('utf-8')
    if methodB in listOfMethodOptions :
        setMethod(methodB, options.encode('utf-8'))
    else :
        raise MethodError("This method doesn't exist, please see listOfMethodOptions for selecting a method")

def PyInitMethod() :
    initMethod()

def PyGetInitime() :
    return getInitime()

def PyRunMethod(g, h) :
    runMethod(g,h)

def PyGetUpperBound(g,h) :
    return getUpperBound(g,h)

def PyGetLowerBound(g,h) :
    return getLowerBound(g,h)

def PyGetAllMap(g,h) :
    return getAllMap(g,h)

def PyGetRuntime(g,h) :
    return getRuntime(g,h)

def PyQuasimetricCost() :
    return quasimetricCosts()

###########################################
##LISTES DES METHODES ET FONCTION DE COUT##
###########################################

listOfEditCostOptions = PyGetEditCostOptions()
listOfMethodOptions = PyGetMethodOptions()
listOfInitOptions = PyGetInitOptions()


########################
##GESTION DES ERREURS ##
########################

class Error(Exception):
    pass

class EditCostError(Error) :
    def __init__(self, message):
        self.message = message
    
class MethodError(Error) :
    def __init__(self, message):
        self.message = message


##############################
##FONCTIONS PYTHON DE CALCUL##
##############################

    
def computeEditDistanceOnGXlGraphs(pathFolder, pathXML, editCost, method, options) :

    PyRestartEnv()
    
    PyLoadGXLGraph(pathFolder, pathXML)
    listID = PyGetGraphIds()
    print("Number of graphs = " + str(len(listID)))

    PySetEditCost(editCost)
    PyInitEnv()
    
    PySetMethod(method, options)
    PyInitMethod()

    res = []
    for g in listID :
        for h in listID :
            PyRunMethod(g,h)
            res.append((PyGetUpperBound(g,h), PyGetRuntime(g,h)))
            
    return res

    
