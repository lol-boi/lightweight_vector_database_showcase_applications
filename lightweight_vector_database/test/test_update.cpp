#include "database.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>

// Helper to check if a vector of results contains a specific ID
bool results_contain_id(const std::vector<hnsw::QueryResult>& results, uint32_t id) {
    return std::any_of(results.begin(), results.end(), [id](const hnsw::QueryResult& r) {
        return r.id == id;
    });
}

void test_update() {
    std::cout << "--- Running test_update ---" << std::endl;
    hnsw::Database db("update_test_db.bin", 2);

    // 1. Insert initial vectors
    uint32_t id0 = db.insert({1.0f, 1.0f}); // ID 0
    uint32_t id1 = db.insert({2.0f, 2.0f}); // ID 1
    assert(id0 == 0);
    assert(id1 == 1);
    std::cout << "Step 1 passed: Initial vectors inserted." << std::endl;

    // 2. Update vector with ID 0
    std::vector<float> new_vec = {1.5f, 1.5f};
    uint32_t new_id = db.update_vector(id0, new_vec, {{"status", "updated"}});
    
    // The new ID should be 2, as it's the next available ID
    assert(new_id == 2); 
    std::cout << "Step 2 passed: Vector updated, new ID is " << new_id << "." << std::endl;

    // 3. Query the database
    std::vector<hnsw::QueryResult> results = db.query({1.0f, 1.0f}, 3, nullptr, {hnsw::Include::ID, hnsw::Include::METADATA});
    
    std::cout << "Query results IDs: ";
    for (const auto& res : results) {
        std::cout << res.id << " ";
    }
    std::cout << std::endl;

    // 4. Verify the results
    // The old ID 0 should not be present
    assert(!results_contain_id(results, id0)); 
    // The original ID 1 should still be present
    assert(results_contain_id(results, id1));
    // The new ID (2) for the updated vector should be present
    assert(results_contain_id(results, new_id));

    // Check the metadata of the updated vector
    for (const auto& res : results) {
        if (res.id == new_id) {
            assert(res.metadata.at("status") == "updated");
        }
    }
    std::cout << "Step 3 passed: Query results are correct after update." << std::endl;

    std::cout << "test_update passed." << std::endl;
}


int main() {
    test_update();

    std::cout << "\nAll update tests passed!" << std::endl;

    // Clean up db file
    std::remove("update_test_db.bin");

    return 0;
}
