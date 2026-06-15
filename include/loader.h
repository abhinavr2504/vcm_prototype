#pragma once
#include <string>
#include "config.h"
#include "partition.h"
#include "dataset.h"

// Reads a graph file in edge-list format:
//   # comment
//   <num_vertices> <num_edges>
//   <src> <dst>
//   ...
// Builds a CSR-format graph, initialises hidden states, and activates all
// vertices in the initial frontier.
Partition load_partition(const std::string& path, const RunConfig& cfg);

// Builds a partition directly from a loaded Dataset (E5B path).
Partition load_partition_from_dataset(const Dataset& dataset, const RunConfig& cfg);
