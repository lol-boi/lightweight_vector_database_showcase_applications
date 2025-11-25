#include "database.h"
#include <iostream>
#include <vector>
#include <cassert>

void test_sq_quantization() {
    std::cout << "Running test_sq_quantization..." << std::endl;

    const std::string db_path = "test_sq.db";
    const size_t vector_dimension = 4;
    const int M = 16;
    const int efConstruction = 200;
    const int efSearch = 50;

    // Create a database with SQ enabled
    hnsw::Database db(db_path, vector_dimension, M, efConstruction, efSearch, hnsw::DistanceMetric::L2, false, 0, true);

    // Insert some vectors
    std::vector<std::vector<float>> vectors_to_insert = {
        {1.0, 1.0, 1.0, 1.0},
        {1.1, 1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0, 2.0},
        {2.1, 2.0, 2.0, 2.0},
        {3.0, 3.0, 3.0, 3.0},
        {3.1, 3.0, 3.0, 3.0}
    };
    for (const auto& vec : vectors_to_insert) {
        db.insert(vec);
    }

    // Train the quantizer and rebuild the index
    db.rebuild_index();

    // Query for a vector
    std::vector<float> query_vec = {1.0, 1.0, 1.0, 1.0};
    auto results = db.query(query_vec, 2);

    // Basic assertions
    assert(results.size() == 2);
    assert(results[0].id == 0 || results[0].id == 1);
    assert(results[1].id == 0 || results[1].id == 1);

    std::cout << "Query results are reasonable." << std::endl;

    // Save and load the database
    db.save();
    hnsw::Database loaded_db(db_path, vector_dimension, M, efConstruction, efSearch, hnsw::DistanceMetric::L2, true, 0, true);
    
    auto loaded_results = loaded_db.query(query_vec, 2);
    assert(loaded_results.size() == 2);
    assert(loaded_results[0].id == results[0].id);
    assert(loaded_results[1].id == results[1].id);

    std::cout << "Save and load with SQ works." << std::endl;

    // Clean up
    remove(db_path.c_str());

    std::cout << "test_sq_quantization passed." << std::endl;
}

int main() {
    test_sq_quantization();
    return 0;
}
