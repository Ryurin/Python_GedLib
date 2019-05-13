#include <iostream>
#include "essai.h"
#include "../include/gedlib-master/src/env/ged_env.hpp"

using namespace std;

//ged::GEDEnv<UserNodeID, UserNodeLabel, UserEdgeLabel> env;
ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> env;
//ged::GEDEnv<int, double, double> env2;

//typedef std::map<std::string, std::string> GXLLabel;
//typedef std::string GXLNodeID;

bool initialized = false;

bool isInitialized(){
	return initialized;
}

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

std::vector<std::string> initStringOptions = { 
	"LAZY_WITHOUT_SHUFFLED_COPIES", 
	"EAGER_WITHOUT_SHUFFLED_COPIES", 
	"LAZY_WITH_SHUFFLED_COPIES", 
	"EAGER_WITH_SHUFFLED_COPIES"
};

std::vector<std::string> getInitStringOptions(){
	return initStringOptions;
}
 
std::map<std::string, ged::Options::InitType> initOptions = {
	{"LAZY_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITHOUT_SHUFFLED_COPIES},
	{"EAGER_WITHOUT_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITHOUT_SHUFFLED_COPIES},
	{"LAZY_WITH_SHUFFLED_COPIES", ged::Options::InitType::LAZY_WITH_SHUFFLED_COPIES},
	{"EAGER_WITH_SHUFFLED_COPIES", ged::Options::InitType::EAGER_WITH_SHUFFLED_COPIES}
};

void loadGXLGraph(std::string pathFolder, std::string pathXML){
	 std::vector<ged::GEDGraph::GraphID> tmp_graph_ids(env.load_gxl_graphs(pathFolder, pathXML));
}

std::vector<std::size_t> getGraphIds(){
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
	return std::stoi(std::to_string(newId));
}

//void add_node(GEDGraph::GraphID graph_id, const UserNodeID & node_id, const UserNodeLabel & node_label);
//void add_edge(GEDGraph::GraphID graph_id, const UserNodeID & tail, const UserNodeID & head, const UserEdgeLabel & edge_label, bool ignore_duplicates = true);

void addNode(std::size_t graphId, ged::GXLNodeID nodeId, ged::GXLLabel nodeLabel){
	env.add_node(graphId, nodeId, nodeLabel);
}

/*void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel){
	env.add_edge(graphId, tail, head, edgeLabel);
}*/

void addEdge(std::size_t graphId, ged::GXLNodeID tail, ged::GXLNodeID head, ged::GXLLabel edgeLabel, bool ignoreDuplicates){
	env.add_edge(graphId, tail, head, edgeLabel, ignoreDuplicates);
}

void restartEnv(){
	env = ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel>();
	initialized = false;
}

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

/*void initEnv(){
	env.init();
	initialized = true;
}*/

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

/*std::vector<long unsigned int> getForwardMap(int g, int h){
	return env.get_node_map(g,h).get_forward_map(); 
}

std::vector<long unsigned int> getBackwardMap(int g, int h){
	return env.get_node_map(g,h).get_backward_map(); 
}*/

std::vector<std::vector<unsigned long int>> getAllMap(std::size_t g, std::size_t h){
	ged::NodeMap pika = env.get_node_map(g,h);
	std::vector<unsigned long int> rondou = pika.get_forward_map(); 
	std::vector<unsigned long int> grodou = pika.get_backward_map(); 
	std::vector<std::vector<unsigned long int>> res; 
	res.push_back(rondou);
	res.push_back(grodou); 
	return res;
}

double getRuntime(std::size_t g, std::size_t h){
	return env.get_runtime(g, h);
}

bool quasimetricCosts(){
	return env.quasimetric_costs();
}

std::string toStringVectorInt(std::vector<int> vector){
	std::string res = "";

    for (int i = 0; i != vector.size(); i++)
    {
       res += std::to_string(vector[i]) + " ";
    }
    
    return res;
}

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
	
	loadGXLGraph("include/gedlib-master/data/datasets/Mutagenicity/data/", "collections/MUTA_10.xml");
	std::vector<std::size_t> listIdInt = getGraphIds();
	std::string truc = "";

    for (int i = 0; i != listIdInt.size(); i++)
    {
        truc += std::to_string(listIdInt[i]) + " ";
    }

    std::cout << "Number of graphs = " << listIdInt.size() << ", list of IDs = " << truc << "\n";
    cout << env.graph_ids().first << ", " << env.graph_ids().second << endl;
	setEditCost("CHEM_1");
	initEnv();
	setMethod("truc","");
	initMethod();
	int g = listIdInt[0];
	int h = listIdInt[1];
	runMethod(g,h);
	/*ged::NodeMap pika = env.get_node_map(g,h);
	std::vector<long unsigned int> rondou = pika.get_forward_map(); 
	std::vector<long unsigned int> grodou = pika.get_backward_map(); */
	
	std::vector<std::vector<unsigned long int>> pichu = getAllMap(g,h);
	
	std::string machin = "";

    for (int i = 0; i != pichu[0].size(); i++)
    {
        machin += std::to_string(pichu[0][i]) + " ";
    }
    
    std::string chose = "";
    
    for (int i = 0; i != pichu[1].size(); i++)
    {
        chose += std::to_string(pichu[1][i]) + " ";
    }
	
	std::cout << "\nupper bound = " << getUpperBound(g, h) << ", matrix = " << env.get_node_map(g,h) << ", runtime = " << getRuntime(g, h) << "\n";
	std::cout << "forward map = " << machin << ", backward map = " << chose << "\n";

    return 0;
}

