#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/Python_GedLib/lib/fann/:/export/home/lambertn/Documents/Python_GedLib/lib/libsvm.3.22:/export/home/lambertn/Documents/Python_GedLib/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
import librariesImport
import PythonGedLib


def testAddGraph() :
    PythonGedLib.PyRestartEnv()
    PythonGedLib.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')

    currentID = PythonGedLib.PyAddGraph();
    print(currentID)
    
    PythonGedLib.PyAddNode(currentID, "_1", {"chem" : "C"})
    PythonGedLib.PyAddNode(currentID, "_2", {"chem" : "O"})
    PythonGedLib.PyAddEdge(currentID,"_1", "_2",  {"valence": "1"} );

    listID = PythonGedLib.PyGetAllGraphIds()
    print(listID)
    print(PythonGedLib.PyGetGraphNodeLabels(10))
    print(PythonGedLib.PyGetGraphEdges(10))
    
    for i in listID : 
        print(PythonGedLib.PyGetGraphNodeLabels(i))
        print(PythonGedLib.PyGetGraphEdges(i))

#testAddGraph()

def afficheMatrix(mat) :
    for i in mat :
        line = ""
        for j in i :
            line+=str(j)
            line+=" "
        print(line)

def minitest() :
    PythonGedLib.PyRestartEnv()
    
    print("Here is the mini Python function !")
    
    PythonGedLib.PyLoadGXLGraph("include/gedlib-master/data/datasets/Mutagenicity/data/", "include/gedlib-master/data/collections/Mutagenicity.xml")
    listID = PythonGedLib.PyGetAllGraphIds()
    PythonGedLib.PySetEditCost("CHEM_1")

    PythonGedLib.PyInitEnv()

    PythonGedLib.PySetMethod("BIPARTITE", "")
    PythonGedLib.PyInitMethod()

    g = listID[0]
    h = listID[1]

    PythonGedLib.PyRunMethod(g,h)

    print("Node Map : ", PythonGedLib.PyGetNodeMap(g,h))
    print("Assignment Matrix : ")
    afficheMatrix(PythonGedLib.PyGetAssignmentMatrix(g,h))
    print ("Upper Bound = " + str(PythonGedLib.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(PythonGedLib.PyGetLowerBound(g,h)) + ", Runtime = " + str(PythonGedLib.PyGetRuntime(g,h)))

#minitest()

def test() :
    PythonGedLib.appel()
    
    PythonGedLib.PyRestartEnv()
    
    print("Here is the Python function !")
    
    print("List of Edit Cost Options : ")
    for i in PythonGedLib.listOfEditCostOptions :
        print (i)
    print("")

    print("List of Method Options : ")
    for j in PythonGedLib.listOfMethodOptions :
        print (j)
    print("")

    print("List of Init Options : ")
    for k in PythonGedLib.listOfInitOptions :
        print (k)
    print("")
    
    PythonGedLib.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
    listID = PythonGedLib.PyGetAllGraphIds()
    
    afficheId = ""
    for i in listID :
        afficheId+=str(i) + " "
    print("Number of graphs = " + str(len(listID)) + ", list of Ids = " + afficheId)

    PythonGedLib.PySetEditCost("CHEM_1")

    PythonGedLib.PyInitEnv()

    PythonGedLib.PySetMethod("IPFP", "")
    PythonGedLib.PyInitMethod()

    g = listID[0]
    h = listID[0]

    PythonGedLib.PyRunMethod(g,h)
    liste = PythonGedLib.PyGetAllMap(g,h)
    print("Forward map : " ,PythonGedLib.PyGetForwardMap(g,h), ", Backward map : ", PythonGedLib.PyGetBackwardMap(g,h))
    print("Node Map : ", PythonGedLib.PyGetNodeMap(g,h))
    print ("Upper Bound = " + str(PythonGedLib.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(PythonGedLib.PyGetLowerBound(g,h)) + ", Runtime = " + str(PythonGedLib.PyGetRuntime(g,h)))


test()

