/*
 * RoundRobinScheduler.cpp
 *
 *  Created on: Jun 29, 2012
 *      Author: tcpan
 */

#include "RoundRobinScheduler.h"

namespace cci {
namespace rt {

RoundRobinScheduler::RoundRobinScheduler(std::vector<int> &_roots, std::vector<int> &_leaves) :
	Scheduler_I(_roots, _leaves) {
	rootIdx = rand() % roots.size();
	leafIdx = rand() % leaves.size();
}

RoundRobinScheduler::RoundRobinScheduler(bool _root, bool _leaf) :
	Scheduler_I(_root, _leaf) {
	rootIdx = rand() % roots.size();
	leafIdx = rand() % leaves.size();
}

int RoundRobinScheduler::getRootFromLeaf(int leafId) {
//	std::ostream_iterator<int> os(std::cout, ",");
//	copy(roots.begin(), roots.end(), os);
//	std::cout << std::endl;

	int size = roots.size();
	if (size == 0) return -1;
	else if (size == 1) return roots[0];
	else {
		rootIdx = (rootIdx + 1) % size;
		return roots[rootIdx];
	}
}
int RoundRobinScheduler::getLeafFromRoot(int rootId) {
//	std::ostream_iterator<int> os(std::cout, ",");
//	copy(leaves.begin(), leaves.end(), os);
//	std::cout << std::endl;

	int size = leaves.size();
	if (size == 0) return -1;
	else if (size == 1) return leaves[0];
	else {
		leafIdx = (rootIdx + 1) % size;
		return leaves[leafIdx];
	}
}

// do not maintain position. after removal
int RoundRobinScheduler::removeRoot(int id) {

	int count = Scheduler_I::removeRoot(id);
	// reset the iterators
	if (count == 0) rootIdx = -1;
	else if (count == 1) rootIdx = 0;
	else rootIdx = rand() % count;

	return count;
}

// do not maintain position. after removal
int RoundRobinScheduler::removeLeaf(int id) {

	int count = Scheduler_I::removeLeaf(id);

	// reset the iterators
	if (count == 0) leafIdx = -1;
	else if (count == 1) leafIdx = 0;
	else leafIdx = rand() % count;

	return count;
}
// do not maintain position. after removal
int RoundRobinScheduler::addRoot(int id) {

	int count = Scheduler_I::addRoot(id);

	// reset the iterators
	if (count == 0) rootIdx = -1;
	else if (count == 1) rootIdx = 0;
	else rootIdx = rand() % count;

	return count;
}
// do not maintain position. after removal
int RoundRobinScheduler::addLeaf(int id) {

	int count = Scheduler_I::addLeaf(id);

	// reset the iterators
	if (count == 0) leafIdx = -1;
	else if (count == 1) leafIdx = 0;
	else leafIdx = rand() % count;

	return count;
}


} /* namespace rt */
} /* namespace cci */