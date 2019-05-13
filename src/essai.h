//On ne peut pas inclure ged_env dans le .h
//#include "../include/gedlib-master/src/env/ged_env.hpp"
#include <string>
#include <vector>

//On suppose que les GraphID sont des int
//Les différents enums seront des string de la méthode, transmise par le Python

std::vector<std::string> getEditCostStringOptions();
std::vector<std::string> getMethodStringOptions();
std::vector<std::string> getInitStringOptions();

bool isInitialized();

int appelle();

void restartEnv();

void loadGXLGraph(std::string pathFolder, std::string pathXML);
std::vector<std::size_t> getGraphIds();
std::string getGraphClass(std::size_t id);
std::string getGraphName(std::size_t id);

std::size_t addGraph(std::string name, std::string classe);

void setEditCost(std::string editCost);

/*void initEnv();*/
void initEnv(std::string initOption);

void setMethod(std::string method, std::string options);
void initMethod();
double getInitime();

void runMethod(std::size_t g, std::size_t h );

double getUpperBound(std::size_t g, std::size_t h);
double getLowerBound(std::size_t g,std::size_t h);
std::vector<std::vector<unsigned long int>> getAllMap(std::size_t g, std::size_t h);
double getRuntime(std::size_t g, std::size_t h);
bool quasimetricCosts();
