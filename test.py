#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/Cython_GedLib_2/lib/fann/:/export/home/lambertn/Documents/Cython_GedLib_2/lib/libsvm.3.22:/export/home/lambertn/Documents/Cython_GedLib_2/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
#Permet de fonctionner sur Idle et autre sans définir à chaque fois la variable d'environnement
#os.environ ne fonctionne pas dans ce cas
import librariesImport

import script

#linlin.jia@insa-rouen.fr

truc = script.computeEditDistanceOnGXlGraphs('include/gedlib-master/data/datasets/Mutagenicity/data/','collections/MUTA_10.xml',"CHEM_1", "BIPARTITE", "") 
print(truc)
script.PyRestartEnv()
#script.appel()

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
    listID = script.PyGetGraphIds()
    
    afficheId = ""
    for i in listID :
        afficheId+=str(i) + " "
    print("Number of graphs = " + str(len(listID)) + ", list of Ids = " + afficheId)

    script.PySetEditCost("CHEM_1")

    script.PyInitEnv()

    script.PySetMethod("BIPARTITE", "")
    script.PyInitMethod()

    g = listID[0]
    h = listID[1]

    script.PyRunMethod(g,h)
    liste = script.PyGetAllMap(g,h)
    print("Forward map : " ,script.PyGetForwardMap(g,h), ", Backward map : ", script.PyGetBackwardMap(g,h))
    print ("Upper Bound = " + str(script.PyGetUpperBound(g,h)) + ", Lower Bound = " + str(script.PyGetLowerBound(g,h)) + ", Runtime = " + str(script.PyGetRuntime(g,h)))


test()
