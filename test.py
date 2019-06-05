#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/Cython_GedLib_2/lib/fann/:/export/home/lambertn/Documents/Cython_GedLib_2/lib/libsvm.3.22:/export/home/lambertn/Documents/Cython_GedLib_2/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
#Permet de fonctionner sur Idle et autre sans définir à chaque fois la variable d'environnement
#os.environ ne fonctionne pas dans ce cas
import librariesImport

import script

#linlin.jia@insa-rouen.fr

#truc = script.computeEditDistanceOnGXlGraphs('include/gedlib-master/data/datasets/alkane/','include/gedlib-master/data/collections/alkane.xml',"CHEM_1", "IPFP", "") 
#print(truc)
#script.PyRestartEnv()
#script.appel()
#script.PyGetGraphInternalId(0)

def testAddGraph() :
    script.PyRestartEnv()
    #script.PyLoadGXLGraph('include/gedlib-master/data/datasets/alkane/','include/gedlib-master/data/collections/alkane.xml')
    script.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')

    currentID = script.PyAddGraph();
    print(currentID)
    script.PyAddNode(currentID, "_1", {"chem" : "C"})
    script.PyAddNode(currentID, "_2", {"chem" : "O"})
    script.PyAddEdge(currentID,"_1", "_2",  {"valence": "1"} );
    #script.PySetEditCost("CHEM_1")
    listID = script.PyGetAllGraphIds()
    print(listID)
    print(script.PyGetGraphNodeLabels(10))
    print(script.PyGetGraphEdges(10))
    
    #print(script.PyGetGraphName(7))
    for i in listID : 
        print(script.PyGetGraphNodeLabels(i))
        print(script.PyGetGraphEdges(i))
    #print(script.PyGetGraphEdges(7))

#testAddGraph()

##<graph id="molecule_3486" edgeids="true" edgemode="undirected">
##<node id="1"><attr name="chem"><string>C</string></attr></node>
##<node id="2"><attr name="chem"><string>C</string></attr></node>
##<node id="3"><attr name="chem"><string>C</string></attr></node>
##<node id="4"><attr name="chem"><string>O</string></attr></node>
##<node id="5"><attr name="chem"><string>Cl</string></attr></node>
##<node id="6"><attr name="chem"><string>Cl</string></attr></node>
##<node id="7"><attr name="chem"><string>Cl</string></attr></node>
##<node id="8"><attr name="chem"><string>Cl</string></attr></node>
##<node id="9"><attr name="chem"><string>H</string></attr></node>
##<node id="10"><attr name="chem"><string>H</string></attr></node>
##<edge from="1" to="2"><attr name="valence"><int>1</int></attr></edge>
##<edge from="1" to="3"><attr name="valence"><int>1</int></attr></edge>
##<edge from="1" to="4"><attr name="valence"><int>2</int></attr></edge>
##<edge from="2" to="5"><attr name="valence"><int>1</int></attr></edge>
##<edge from="2" to="6"><attr name="valence"><int>1</int></attr></edge>
##<edge from="3" to="7"><attr name="valence"><int>1</int></attr></edge>
##<edge from="3" to="8"><attr name="valence"><int>1</int></attr></edge>
##<edge from="2" to="9"><attr name="valence"><int>1</int></attr></edge>
##<edge from="3" to="10"><attr name="valence"><int>1</int></attr></edge>
##</graph>

def recuptest() :
    print("Here is the recuperation Python function !")
    listID = script.PyGetAllGraphIds()
    g = listID[2]
    h = listID[3]

    print("Forward map : " ,script.PyGetForwardMap(g,h), ", Backward map : ", script.PyGetBackwardMap(g,h))
    print("Node Map : ", script.PyGetNodeMap(g,h))
    print ("Upper Bound = " + str(script.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(script.PyGetLowerBound(g,h)) + ", Runtime = " + str(script.PyGetRuntime(g,h)))

#recuptest()

def afficheMatrix(mat) :
    for i in mat :
        line = ""
        for j in i :
            line+=str(j)
            line+=" "
        print(line)

def minitest() :
    script.PyRestartEnv()
    
    print("Here is the mini Python function !")
    
    #script.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
    script.PyLoadGXLGraph("include/gedlib-master/data/datasets/Mutagenicity/data/", "include/gedlib-master/data/collections/Mutagenicity.xml")
    listID = script.PyGetAllGraphIds()
    
##    afficheId = ""
##    for i in listID :
##        afficheId+=str(i) + " "
##    print("Number of graphs = " + str(len(listID)) + ", list of Ids = " + afficheId)

    script.PySetEditCost("CHEM_1")

    script.PyInitEnv()

    script.PySetMethod("BIPARTITE", "")
    script.PyInitMethod()

    g = listID[0]
    h = listID[1]

    script.PyRunMethod(g,h)
    #print("Forward map : " ,script.PyGetForwardMap(g,h), ", Backward map : ", script.PyGetBackwardMap(g,h))
    print("Node Map : ", script.PyGetNodeMap(g,h))
    print("Assignment Matrix : ")
    afficheMatrix(script.PyGetAssignmentMatrix(g,h))
    print ("Upper Bound = " + str(script.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(script.PyGetLowerBound(g,h)) + ", Runtime = " + str(script.PyGetRuntime(g,h)))

#minitest()

def test() :
    script.appel()
    
    script.PyRestartEnv()
    
    print("Here is the Python function !")
    
    print("List of Edit Cost Options : ")
    for i in script.listOfEditCostOptions :
        print (i)
    print("")

    print("List of Method Options : ")
    for j in script.listOfMethodOptions :
        print (j)
    print("")

    print("List of Init Options : ")
    for k in script.listOfInitOptions :
        print (k)
    print("")
    
    script.PyLoadGXLGraph('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
    listID = script.PyGetAllGraphIds()
    
    afficheId = ""
    for i in listID :
        afficheId+=str(i) + " "
    print("Number of graphs = " + str(len(listID)) + ", list of Ids = " + afficheId)

    script.PySetEditCost("CHEM_1")

    script.PyInitEnv()

    script.PySetMethod("IPFP", "")
    script.PyInitMethod()

    g = listID[0]
    h = listID[0]

    script.PyRunMethod(g,h)
    liste = script.PyGetAllMap(g,h)
    print("Forward map : " ,script.PyGetForwardMap(g,h), ", Backward map : ", script.PyGetBackwardMap(g,h))
    print("Node Map : ", script.PyGetNodeMap(g,h))
    print ("Upper Bound = " + str(script.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(script.PyGetLowerBound(g,h)) + ", Runtime = " + str(script.PyGetRuntime(g,h)))


test()

runtime = 0.000883781
Runtime = 0.000154826

