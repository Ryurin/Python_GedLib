/****************************************************************************
 *                                                                          *
 *   Copyright (C) 2019 by Natacha Lambert and David B. Blumenthal          *
 *                                                                          *
 *   This file should be used by Python.                                    *
 * 	 Please call the Python module if you want to use GedLib with this code.* 
 *                                                                          *
 * 	 Otherwise, you can directly use GedLib for C++.                        *
 *                                                                          *
 ***************************************************************************/
 
/*!
 * @file essai.cpp
 * @brief Functions definition to call easly GebLib in Python without Gedlib's types
 */

//Include standard libraries + GedLib library
#include <iostream>
#include "essai.h"
#include "../include/gedlib-master/src/env/ged_env.hpp"

using namespace std;

//Definition of types and templates used in this code for my human's memory :). 
//ged::GEDEnv<UserNodeID, UserNodeLabel, UserEdgeLabel> env;
//template<class UserNodeID, class UserNodeLabel, class UserEdgeLabel> struct ExchangeGraph

//typedef std::map<std::string, std::string> GXLLabel;
//typedef std::string GXLNodeID;

ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> env; //Environment variable


bool initialized = false; //Initialization boolean (because Env has one but not accessible). 

bool isInitialized(){
	return initialized;
}

//!< List of available edit cost functions readable by Python.  
std::vector<std::string> editCostStringOptions = { 
	"CHEM_1", 
	"CHEM_2", 
	"CMU", 
	"GREC_1", 
	"GREC_2", 
	"LETTER", 
	"FINGERPRINT", 
	"PROTEIN", 
	"CONSTANT" 
};

std::vector<std::string> getEditCostStringOptions(){
	return editCostStringOptions;
}

//!< Map of available edit cost functions between enum type in C++ and string in Python  
std::map<std::string, ged::Options::EditCosts> editCostOptions = {
	{"CHEM_1", ged::Options::EditCosts::CHEM_1},
	{"CHEM_2", ged::Options::EditCosts::CHEM_2},
	{"CMU", ged::Options::EditCosts::CMU},
	{"GREC_1", ged::Options::EditCosts::GREC_1},
	{"GREC_2", ged::Options::EditCosts::GREC_2},
	{"LETTER", ged::Options::EditCosts::LETTER},
	{"FINGERPRINT", ged::Options::EditCosts::FINGERPRINT},
	{"PROTEIN", ged::Options::EditCosts::PROTEIN},
	{"CONSTANT", ged::Options::EditCosts::CONSTANT}	
};

 //!< List of available computation methods readable by Python.  
std::vector<std::string> methodStringOptions = { 
	"BRANCH",
	"BRANCH_FAST",
	"BRANCH_TIGHT",
	"BRANCH_UNIFORM",
	"BRANCH_COMPACT",
	"PARTITION",
	"HYBRID",
	"RING",
	"ANCHOR_AWARE_GED",
	"WALKS",
	"IPFP",
	"BIPARTITE",
	"SUBGRAPH",
	"NODE",
	"RING_ML",
	"BIPARTITE_ML",
	"REFINE",
	"BP_BEAM",
	"SIMULATED_ANNEALING",
	"HED",
	"STAR"				 
}; 

std::vector<std::string> getMethodStringOptions(){
	return methodStringOptions;
}

//!< Map of available computation methods readables between enum type in C++ and string in Python  
std::map<std::string, ged::Options::GEDMethod> methodOptions = {
	{"BRANCH", ged::Options::GEDMethod::BRANCH},
	{"BRANCH_FAST", ged::Options::GEDMethod::BRANCH_FAST},
	{"BRANCH_TIGHT", ged::Options::GEDMethod::BRANCH_TIGHT},
	{"BRANCH_UNIFORM", ged::Options::GEDMethod::BRANCH_UNIFORM},
	{"BRANCH_COMPACT", ged::Options::GEDMethod::BRANCH_COMPACT},
	{"PARTITION", ged::Options::GEDMethod::PARTITION},
	{"HYBRID", ged::Options::GEDMethod::HYBRID},
	{"RING", ged::Options::GEDMethod::RING},
	{"ANCHOR_AWARE_GED", ged::Options::GEDMethod::ANCHOR_AWARE_GED},
	{"WALKS", ged::Options::GEDMethod::WALKS},
	{"IPFP", ged::Options::GEDMethod::IPFP},
	{"BIPARTITE", ged::Options::GEDMethod::BIPARTITE},
	{"SUBGRAPH", ged::Options::GEDMethod::SUBGRAPH},
	{"NODE", ged::Options::GEDMethod::NODE},
	{"RING_ML", ged::Options::GEDMethod::RING_ML},
	{"BIPARTITE_ML",ged::Options::GEDMethod::BIPARTITE_ML},
	{"REFINE",ged::Options::GEDMethod::REFINE},
	{"BP_BEAM", ged::Options::GEDMethod::BP_BEAM},
	{"SIMULATED_ANNEALING", ged::Options::GEDMethod::SIMULATED_ANNEALING},
	{"HED", ged::Options::GEDMethod::HED},
	{"STAR"	, ged::Options::GEDMethod::STAR},	
};

//!<List of available initilaization options readable by Python.
std::vector<std::string> initStringOptions = { 
	"LAZY_WITHOUT_SHUFFLED_COPIES", 
	"EAGER_WITHOUT_SHUFFLED_COPIES", 
	"LAZY_WITH_SHUFFLED_COPIES", 
	"EAGER_WITH_SHUFFLED_COPIES"
};

std::vector<std::string> getInitStringOptions(){
	return initStringOptions;
}

//!< Map of available initilaization options readables between enum type in C++ and string in Python 
std::map<std::string, ged::Options::InitType> initOptions = {
	{"LAZY_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITHOUT_SHUFFLED_COPIES},
	{"EAGER_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITHOUT_SHUFFLED_COPIES},
	{"LAZY_WITH_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITH_SHUFFLED_COPIES},
	{"EAGER_WITH_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITH_SHUFFLED_COPIES}
};

void restartEnv(){
	env = ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	initialized = false;
}

void loadGXLGraph(std::string pathFolder, std::string pathXML){
	 std::vector<ged::GEDGraph::GraphID> tmp_graph_ids(env.load_gxl_graphs(pathFolder, pathXML));
}

std::pair<std::size_t,std::size_t> getGraphIds(){
	return env.graph_ids();
}

std::vector<std::size_t> getAllGraphIds(){
	std::vector<std::size_t> listID;
	for (std::size_t i = env.graph_ids().first; i != env.graph_ids().second; i++){
		listID.push_back(i);
    }
	return listID;
}

std::string getGraphClass(std::size_t id){
	return env.get_graph_class(id);
}

std::string getGraphName(std::size_t id){
	return env.get_graph_name(id);
}

std::size_t addGraph(std::string name, std::string classe){
	ged::GEDGraph::GraphID newId = env.add_graph(name, classe); 
	initialized = false;
	return std::stoi(std::to_string(newId));
}

void addNode(std::size_t graphId, std::string nodeId, std::map<std::string, std::string> nodeLabel){
	env.add_node(graphId, nodeId, nodeLabel);
	initialized = false;
}

/*void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel){
	env.add_edge(graphId, tail, head, edgeLabel);
}*/

void addEdge(std::size_t graphId, std::string tail, std::string head, std::map<std::string, std::string> edgeLabel, bool ignoreDuplicates){
	env.add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
	initialized = false;
}

void clearGraph(std::size_t graphId){
	env.clear_graph(graphId);
	initialized = false;
}

/*!
 * @brief Returns ged::ExchangeGraph representation.
 * @param graphId ID of the selected graph.
 * @return ged::ExchangeGraph representation of the selected graph.
 */
ged::ExchangeGraph<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> getGraph(std::size_t graphId){
	return env.get_graph(graphId);
}

std::size_t getGraphInternalId(std::size_t graphId){
	return getGraph(graphId).id;
}

std::size_t getGraphNumNodes(std::size_t graphId){
	return getGraph(graphId).num_nodes;
}

std::size_t getGraphNumEdges(std::size_t graphId){
	return getGraph(graphId).num_edges;
}

std::vector<std::string> getGraphOriginalNodeIds(std::size_t graphId){
	return getGraph(graphId).original_node_ids;
}

std::vector<std::map<std::string, std::string>> getGraphNodeLabels(std::size_t graphId){
	return getGraph(graphId).node_labels;
}

std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::map<std::string, std::string>>> getGraphEdges(std::size_t graphId){
	return getGraph(graphId).edges;
}

std::vector<std::list<std::pair<std::size_t, std::map<std::string, std::string>>>> getGraphAdjacenceList(std::size_t graphId){
	return getGraph(graphId).adj_list;
}

/*!
 * @brief Returns the enum EditCost which correspond to the string parameter
 * @param editCost Select one of the predefined edit costs in the list.
 * @return The edit cost function which correspond in the edit cost functions map. 
 */
ged::Options::EditCosts translateEditCost(std::string editCost){
	 for (int i = 0; i != editCostStringOptions.size(); i++){
		 if (editCostStringOptions[i] == editCost){
			 return editCostOptions[editCostStringOptions[i]];
		 } 
	 }
	 return ged::Options::EditCosts::CONSTANT;
}

void setEditCost(std::string editCost){
	env.set_edit_costs(translateEditCost(editCost));
}

void initEnv(){
	env.init();
	initialized = true;
}

/*!
 * @brief Returns the enum IniType which correspond to the string parameter
 * @param initOption Select initialization options.
 * @return The init Type which correspond in the init options map. 
 */
ged::Options::InitType translateInitOptions(std::string initOption){
	 for (int i = 0; i != initStringOptions.size(); i++){
		 if (initStringOptions[i] == initOption){
			 return initOptions[initStringOptions[i]];
		 } 
	 }
	 return ged::Options::InitType::EAGER_WITHOUT_SHUFFLED_COPIES;
}

void initEnv(std::string initOption){
	env.init(translateInitOptions(initOption));
	initialized = true;
}

/*!
 * @brief Returns the enum Method which correspond to the string parameter
 * @param method Select the method that is to be used.
 * @return The computation method which correspond in the edit cost functions map. 
 */
ged::Options::GEDMethod translateMethod(std::string method){
	 for (int i = 0; i != methodStringOptions.size(); i++){
		 if (methodStringOptions[i] == method){
			 return methodOptions[methodStringOptions[i]];
		 } 
	 }
	 return ged::Options::GEDMethod::STAR;
}

void setMethod(std::string method, std::string options){
	env.set_method(translateMethod(method),options);
}

void initMethod(){
	env.init_method();
}

double getInitime(){
	return env.get_init_time();
}

void runMethod(std::size_t g, std::size_t h){
	env.run_method(g, h);
}

double getUpperBound(std::size_t g, std::size_t h){
	return env.get_upper_bound(g, h);
}

double getLowerBound(std::size_t g, std::size_t h){
	return env.get_lower_bound(g, h);
}

std::vector<long unsigned int> getForwardMap(std::size_t g, std::size_t h){
	return env.get_node_map(g,h).get_forward_map(); 
}

std::vector<long unsigned int> getBackwardMap(std::size_t g, std::size_t h){
	return env.get_node_map(g,h).get_backward_map(); 
}

std::size_t getNodeImage(std::size_t g, std::size_t h, std::size_t nodeId){
	if (nodeId < getForwardMap(g,h).size()){
		return env.get_node_map(g,h).image(nodeId);
	}
	else{
		return -1;
	}
}

std::size_t getNodePreImage(std::size_t g, std::size_t h, std::size_t nodeId){
	return env.get_node_map(g,h).pre_image(nodeId);
}

std::vector<pair<std::size_t, std::size_t>> getAdjacenceMatrix(std::size_t g, std::size_t h){
	std::vector<pair<std::size_t, std::size_t>> res; 
	for (int i = 0; i!=getBackwardMap(g,h).size(); i++){
		res.push_back(std::make_pair(i,getNodeImage(g,h,i))); 
	}
	return res;
}

std::vector<std::vector<unsigned long int>> getAllMap(std::size_t g, std::size_t h){
	std::vector<std::vector<unsigned long int>> res; 
	res.push_back(getForwardMap(g, h));
	res.push_back(getBackwardMap(g,h)); 
	return res;
}

double getRuntime(std::size_t g, std::size_t h){
	return env.get_runtime(g, h);
}

bool quasimetricCosts(){
	return env.quasimetric_costs();
}

/*!
 * @brief Returns the string which contains all element of a int list. 
 * @param vector The vector to translate. 
 * @return The string which contains all elements separated with a blank space. 
 */
std::string toStringVectorInt(std::vector<int> vector){
	std::string res = "";

    for (int i = 0; i != vector.size(); i++)
    {
       res += std::to_string(vector[i]) + " ";
    }
    
    return res;
}

/*!
 * @brief Returns the string which contains all element of a unsigned long int list. 
 * @param vector The vector to translate. 
 * @return The string which contains all elements separated with a blank space. 
 */
std::string toStringVectorInt(std::vector<unsigned long int> vector){
	std::string res = "";

    for (int i = 0; i != vector.size(); i++)
    {
        res += std::to_string(vector[i]) + " ";
    }
    
    return res;
}

int appelle()
{
    cout << "Hello world!" << endl;
    cout << "Here is the C++ function !" << endl;

    //On ne peut pas créer l'XML tout seul, les GXL n'ont pas leur classe. Faire un créateur de XML, pourquoi pas, mais il faut que l'utilisateur
    //renseigne certaines choses

    //Chargement des graphes, on utilise un XML pour notifier lesquelles // Graph_ids le seul moyen d'avoir les ID
    //On peut loader à la suite deux univers, juste il faut rassembler la liste des ids je pense
    /*std::vector<ged::GEDGraph::GraphID> graph_ids(env.load_gxl_graphs("include/gedlib-master/data/datasets/Mutagenicity/data/", "collections/MUTA_10.xml"));
    std::vector<ged::GEDGraph::GraphID> graph_ids2(env.load_gxl_graphs("include/gedlib-master/data/datasets/Mutagenicity/data/", "collections/MUTA_102.xml"));

    std::string truc = "";

    graph_ids.insert( graph_ids.end(), graph_ids2.begin(), graph_ids2.end() );

    for (int i = 0; i != graph_ids.size(); i++)
    {
        truc += std::to_string(graph_ids[i]) + " ";
    }

    std::cout << "Number of graphs = " << graph_ids.size() << ", list of IDs = " << truc << "\n";

    //Selon l'option dans l'enum
    env.set_edit_costs(ged::Options::EditCosts::CHEM_1);
    env.init();

    //selon l'enum + string pour les options en mode bash
    env.set_method(ged::Options::GEDMethod::BIPARTITE,"");
    //env.set_method(ged::Options::GEDMethod::BRANCH_TIGHT,"");
    //env.set_method(ged::Options::GEDMethod::REFINE, "--threads 1 --initial-solutions 40 --randomness PSEUDO");
    env.init_method();

    //Pour sélectionner deux graphes
    ged::GEDGraph::GraphID g {graph_ids[0]};
	ged::GEDGraph::GraphID h {graph_ids[1]};

	env.run_method(g, h); //Demande d'ID des graphes
	std::cout << "\nupper bound = " << env.get_upper_bound(g, h) << ", matrix = " << env.get_node_map(g,h) << ", runtime = " << env.get_runtime(g, h) << "\n";*/
	
	loadGXLGraph("include/gedlib-master/data/datasets/Mutagenicity/data/", "/export/home/lambertn/Documents/Cython_GedLib_2/include/gedlib-master/data/collections/Mutagenicity.xml"); //"collections/MUTA_10.xml"
	std::pair<std::size_t, std::size_t> listIdInt = getGraphIds();
	/*std::string truc = "";

    for (int i = listIdInt.first; i != listIdInt.second; i++)
    {
        truc += std::to_string(i) + " ";
    }*/

    std::cout << "Number of graphs = " << listIdInt.second /*<< ", list of IDs = " << truc*/ << "\n";
    //cout << env.graph_ids().first << ", " << env.graph_ids().second << endl;
	setEditCost("CHEM_1");
	initEnv();
	setMethod("BIPARTITE","");
	initMethod();
	std::size_t g = listIdInt.first;
	std::size_t h = listIdInt.first +1;
	runMethod(g,h);
    
    /*for (int i = 0; i!=pichu[0].size(); i++){
		//std::cout << env.get_node_map(g,h).pre_image(pichu[1][i]);
		std::cout << i;
		std::cout << env.get_node_map(g,h).image(i) << "\n";
	}*/
	
	/*std::vector<pair<std::size_t, std::size_t>> kirby = getAdjacenceMatrix(g,h);
	for (int i = 0; i!=kirby.size(); i++){
		std::cout << kirby[i].first << " " << kirby[i].second << "\n";
	}*/
	
	
	std::cout << "\nupper bound = " << getUpperBound(g, h) << ", matrix = " << env.get_node_map(g,h) << ", runtime = " << getRuntime(g, h) << "\n";
	std::cout << "forward map = " << toStringVectorInt(getForwardMap(g,h)) << ", backward map = " << toStringVectorInt(getBackwardMap(g,h)) << "\n\n";

    return 0;
}

