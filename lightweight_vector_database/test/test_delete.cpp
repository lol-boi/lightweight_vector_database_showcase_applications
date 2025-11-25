#include "database.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <string>

// Helper to check if a vector of results contains a specific ID
bool results_contain_id(const std::vector<hnsw::QueryResult>& results, uint32_t id) {
    return std::any_of(results.begin(), results.end(), [id](const hnsw::QueryResult& r) {
        return r.id == id;
    });
}

void test_soft_delete() {
    std::cout << "--- Running test_soft_delete ---" << std::endl;
    hnsw::Database db("delete_test_db.bin", 2);

    db.insert({1.0f, 1.0f}); // ID 0
    db.insert({2.0f, 2.0f}); // ID 1
    db.insert({3.0f, 3.0f}); // ID 2

    // 1. Before deletion, all nodes should be queryable
    std::vector<hnsw::QueryResult> results_before = db.query({1.1f, 1.1f}, 3);
    assert(results_before.size() == 3);
    assert(results_contain_id(results_before, 0));
    assert(results_contain_id(results_before, 1));
    assert(results_contain_id(results_before, 2));
    std::cout << "Step 1 passed: All nodes present before delete." << std::endl;

    // 2. Delete node with ID 1
    db.delete_vector(1);
    std::cout << "Step 2 passed: Deleted vector with ID 1." << std::endl;

    // 3. After deletion, node 1 should not be in results
    std::vector<hnsw::QueryResult> results_after = db.query({1.1f, 1.1f}, 3);
    assert(results_after.size() == 2);
    assert(results_contain_id(results_after, 0));
    assert(!results_contain_id(results_after, 1)); // Key assertion
    assert(results_contain_id(results_after, 2));
    std::cout << "Step 3 passed: Deleted node is not in query results." << std::endl;
    
    std::cout << "test_soft_delete passed." << std::endl;
}

void test_rebuild_index() {
    std::cout << "--- Running test_rebuild_index ---" << std::endl;
    hnsw::Database db("rebuild_test_db.bin", 2);

    db.insert({1.0f, 1.0f}); // ID 0
    db.insert({2.0f, 2.0f}); // ID 1
    db.insert({3.0f, 3.0f}); // ID 2

    // Delete node 1
    db.delete_vector(1);

    // 1. Before rebuild, query works as expected
    std::vector<hnsw::QueryResult> results_before_rebuild = db.query({1.1f, 1.1f}, 3);
    assert(results_before_rebuild.size() == 2);
    assert(!results_contain_id(results_before_rebuild, 1));
    std::cout << "Step 1 passed: Query is correct before rebuild." << std::endl;

    // 2. Rebuild the index
    db.rebuild_index();
    std::cout << "Step 2 passed: Index rebuilt." << std::endl;

    // 3. After rebuild, query should still be correct and IDs should be compacted
    std::vector<hnsw::QueryResult> results_after_rebuild = db.query({1.1f, 1.1f}, 3);
    assert(results_after_rebuild.size() == 2);
    // After deleting ID 1 and rebuilding, the vectors that were ID 0 and 2 are now ID 0 and 1.
    assert(results_contain_id(results_after_rebuild, 0));
    assert(results_contain_id(results_after_rebuild, 1));
    assert(!results_contain_id(results_after_rebuild, 2));
    std::cout << "Step 3 passed: Query is correct after rebuild." << std::endl;

    std::cout << "test_rebuild_index passed." << std::endl;
}

void test_persistence_with_deletes() {
    std::cout << "--- Running test_persistence_with_deletes ---" << std::endl;
    std::string db_path = "persistence_delete_test.bin";

    // 1. Create DB, insert data, delete one vector, and save
    {
        hnsw::Database db(db_path, 2);
        db.insert({1.0f, 1.0f}); // ID 0
        db.insert({2.0f, 2.0f}); // ID 1
        db.insert({3.0f, 3.0f}); // ID 2
        db.delete_vector(1);
        db.save();
        std::cout << "Step 1 passed: DB created, vector deleted, and saved." << std::endl;
    }

    // 2. Load the DB and check if the deleted vector is still gone
    {
        hnsw::Database db(db_path, 2, 16, 200, 50, hnsw::DistanceMetric::L2, true); // Read-only
        std::vector<hnsw::QueryResult> results = db.query({1.1f, 1.1f}, 3);
        assert(results.size() == 2);
        assert(results_contain_id(results, 0));
        assert(!results_contain_id(results, 1));
        assert(results_contain_id(results, 2));
        std::cout << "Step 2 passed: Loaded DB correctly filters deleted vector." << std::endl;
    }

    std::remove(db_path.c_str());
    std::cout << "test_persistence_with_deletes passed." << std::endl;
}


int main() {
    test_soft_delete();
    test_rebuild_index();
    test_persistence_with_deletes();

    std::cout << "\nAll deletion tests passed!" << std::endl;

    // Clean up db files
    std::remove("delete_test_db.bin");
    std::remove("rebuild_test_db.bin");

    return 0;
}
