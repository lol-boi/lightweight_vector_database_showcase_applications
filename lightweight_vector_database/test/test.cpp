#include "database.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath> // For std::abs
#include <stdexcept> // For std::invalid_argument

// Helper to compare floats with tolerance
bool float_equals(float a, float b, float epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

void test_l2_distance_hnsw() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5, hnsw::DistanceMetric::L2); // vector_dimension = 2

    hnsw_graph.insert({0.0f, 0.0f}, {}); // Node 0
    hnsw_graph.insert({1.0f, 0.0f}, {}); // Node 1
    hnsw_graph.insert({0.0f, 1.0f}, {}); // Node 2

    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors({0.1f, 0.1f}, 1);
    assert(results.size() == 1);
    assert(results[0].id == 0); // (0,0) is closest to (0.1,0.1)

    std::cout << "test_l2_distance_hnsw passed." << std::endl;
}

void test_cosine_distance_hnsw() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5, hnsw::DistanceMetric::COSINE); // vector_dimension = 2

    hnsw_graph.insert({1.0f, 0.0f}, {}); // Node 0 (angle 0)
    hnsw_graph.insert({0.0f, 1.0f}, {}); // Node 1 (angle 90)
    hnsw_graph.insert({1.0f, 1.0f}, {}); // Node 2 (angle 45)
    hnsw_graph.insert({-1.0f, 0.0f}, {});// Node 3 (angle 180)

    // Query vector (1, 0.1) - very close to (1,0)
    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors({1.0f, 0.1f}, 1);
    assert(results.size() == 1);
    assert(results[0].id == 0);

    // Query vector (0.1, 1) - very close to (0,1)
    results = hnsw_graph.k_nearest_neighbors({0.1f, 1.0f}, 1);
    assert(results.size() == 1);
    assert(results[0].id == 1);

    // Query vector (1,1) - should be closest to Node 2
    results = hnsw_graph.k_nearest_neighbors({1.0f, 1.0f}, 1);
    assert(results.size() == 1);
    assert(results[0].id == 2);

    std::cout << "test_cosine_distance_hnsw passed." << std::endl;
}

void test_inner_product_distance_hnsw() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5, hnsw::DistanceMetric::IP); // vector_dimension = 2

    hnsw_graph.insert({1.0f, 1.0f}, {}); // Node 0 (IP with (1,1) = 2)
    hnsw_graph.insert({1.0f, 0.0f}, {}); // Node 1 (IP with (1,1) = 1)
    hnsw_graph.insert({-1.0f, -1.0f}, {});// Node 2 (IP with (1,1) = -2)

    // Query vector (1,1). We want max inner product, which means min negative inner product.
    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors({1.0f, 1.0f}, 1);
    assert(results.size() == 1);
    assert(results[0].id == 0); // Node 0 has highest IP (2)

    std::cout << "test_inner_product_distance_hnsw passed." << std::endl;
}

void test_node_structure() {
    hnsw::Node node(10, 3); // ID 10, max_layer 3
    assert(node.id == 10);
    assert(node.max_layer == 3);
    assert(node.neighbors.size() == 4); // Layers 0, 1, 2, 3

    std::cout << "test_node_structure passed." << std::endl;
}

void test_vector_storage() {
    hnsw::VectorStorage storage(2); // vector_dimension = 2
    std::vector<float> v1 = {1.0f, 2.0f};
    std::vector<float> v2 = {3.0f, 4.0f};
    hnsw::Metadata m1 = {{"key", "value1"}};
    hnsw::Metadata m2 = {{"key", "value2"}};

    storage.add_vector(v1, m1);
    storage.add_vector(v2, m2);

    assert(storage.size() == 2);
    assert(storage.get_vector(0) == v1);
    assert(storage.get_vector(1) == v2);
    assert(storage.get_metadata(0) == m1);
    assert(storage.get_metadata(1) == m2);

    std::cout << "test_vector_storage passed." << std::endl;
}

void test_search_layer() {
    hnsw::HNSW hnsw_graph(2); // vector_dimension = 2, Default L2 metric

    // Add vectors
    hnsw::VectorStorage& vector_storage = const_cast<hnsw::VectorStorage&>(hnsw_graph.get_vector_storage());
    vector_storage.add_vector({0.0f, 0.0f}, {}); // Node 0
    vector_storage.add_vector({1.0f, 1.0f}, {}); // Node 1
    vector_storage.add_vector({0.1f, 0.1f}, {}); // Node 2 (closer to 0)
    vector_storage.add_vector({5.0f, 5.0f}, {}); // Node 3 (far from 0)
    vector_storage.add_vector({0.2f, 0.2f}, {}); // Node 4 (closer to 0)

    // Add nodes
    std::vector<hnsw::Node>& nodes = const_cast<std::vector<hnsw::Node>&>(hnsw_graph.get_nodes());
    nodes.emplace_back(0, 0); // Node 0, layer 0
    nodes.emplace_back(1, 0); // Node 1, layer 0
    nodes.emplace_back(2, 0); // Node 2, layer 0
    nodes.emplace_back(3, 0); // Node 3, layer 0
    nodes.emplace_back(4, 0); // Node 4, layer 0

    // Manually set up neighbors for layer 0
    // Node 0 neighbors: 1, 2, 4
    nodes[0].neighbors[0].push_back(1);
    nodes[0].neighbors[0].push_back(2);
    nodes[0].neighbors[0].push_back(4);

    // Node 1 neighbors: 0
    nodes[1].neighbors[0].push_back(0);

    // Node 2 neighbors: 0
    nodes[2].neighbors[0].push_back(0);

    // Node 4 neighbors: 0
    nodes[4].neighbors[0].push_back(0);


    // Query vector
    std::vector<float> query = {0.05f, 0.05f}; // Very close to Node 0, 2, 4

    // Test 1: ef = 1, entry_point = Node 0
    std::vector<int> results1 = hnsw_graph.search_layer(query, 0, 1, 0);
    assert(results1.size() == 1);
    // The closest should be Node 0 (0.0,0.0), Node 2 (0.1,0.1), Node 4 (0.2,0.2)
    // The exact order might depend on tie-breaking, but Node 0 is the entry point and very close.
    // Let's assume Node 0 is returned for ef=1 due to being the entry point and very close.
    // More robust testing would involve checking distances.
    // For now, let's check if one of the very close nodes is returned.
    bool found_close_node = false;
    if (results1[0] == 0 || results1[0] == 2 || results1[0] == 4) {
        found_close_node = true;
    }
    assert(found_close_node);


    // Test 2: ef = 3, entry_point = Node 0
    std::vector<int> results2 = hnsw_graph.search_layer(query, 0, 3, 0);
    assert(results2.size() == 3);
    // Expected closest nodes are 0, 2, 4. Order might vary.
    std::sort(results2.begin(), results2.end());
    assert(results2[0] == 0);
    assert(results2[1] == 2);
    assert(results2[2] == 4);

    std::cout << "test_search_layer passed." << std::endl;
}

void test_full_hnsw_insertion() {
    // Test with M=2, efConstruction=5
    hnsw::HNSW hnsw_graph(2, 2, 5); // vector_dimension = 2, Default L2 metric

    // Insert a few vectors
    hnsw_graph.insert({0.0f, 0.0f}, {}); // Node 0
    hnsw_graph.insert({1.0f, 1.0f}, {}); // Node 1
    hnsw_graph.insert({0.1f, 0.1f}, {}); // Node 2
    hnsw_graph.insert({10.0f, 10.0f}, {});// Node 3
    hnsw_graph.insert({10.1f, 10.1f}, {});// Node 4

    // Assertions to check the state of the graph.
    // These will be simple checks, as the exact structure is non-deterministic.
    assert(hnsw_graph.size() == 5);

    // Check that connections are within M for each layer of each node
    for (const auto& node : hnsw_graph.get_nodes()) {
        for (const auto& layer_neighbors : node.neighbors) {
            assert(layer_neighbors.size() <= 2); // M=2
        }
    }

    // Check that the entry point is the node with the highest layer
    int entry_point_id = hnsw_graph.get_entry_point();
    int max_layer = -1;
    int node_with_max_layer = -1;
    for(const auto& node : hnsw_graph.get_nodes()){
        if(node.max_layer > max_layer){
            max_layer = node.max_layer;
            node_with_max_layer = node.id;
        }
    }
    assert(entry_point_id == node_with_max_layer);

    std::cout << "test_full_hnsw_insertion passed." << std::endl;
}

void test_k_nearest_neighbors() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5); // vector_dimension = 2, Default L2 metric

    // Insert some vectors
    hnsw_graph.insert({0.0f, 0.0f}, {}); // Node 0
    hnsw_graph.insert({1.0f, 1.0f}, {}); // Node 1
    hnsw_graph.insert({0.1f, 0.1f}, {}); // Node 2
    hnsw_graph.insert({0.2f, 0.2f}, {}); // Node 3
    hnsw_graph.insert({10.0f, 10.0f}, {});// Node 4
    hnsw_graph.insert({10.1f, 10.1f}, {});// Node 5

    // Query vector close to (0,0)
    std::vector<float> query = {0.05f, 0.05f};

    // Search for k=3 nearest neighbors
    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors(query, 3);

    assert(results.size() == 3);

    // The expected closest nodes are 0, 2, 3. The order might vary, so sort for comparison.
    std::vector<int> ids;
    for(const auto& r : results) {
        ids.push_back(r.id);
    }
    std::sort(ids.begin(), ids.end());
    assert(ids[0] == 0);
    assert(ids[1] == 2);
    assert(ids[2] == 3);

    std::cout << "test_k_nearest_neighbors passed." << std::endl;
}

void test_vector_dimension_enforcement() {
    hnsw::HNSW hnsw_graph(2); // Initialize with vector_dimension = 2

    // Should insert successfully
    hnsw_graph.insert({1.0f, 2.0f}, {});

    // Should throw an exception for incorrect dimension
    bool caught_exception = false;
    try {
        hnsw_graph.insert({1.0f, 2.0f, 3.0f}, {}); // Dimension 3, expected 2
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        assert(std::string(e.what()) == "Vector dimension mismatch.");
    }
    assert(caught_exception);

    std::cout << "test_vector_dimension_enforcement passed." << std::endl;
}

void test_metadata_filtering() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5); // vector_dimension = 2

    hnsw_graph.insert({0.0f, 0.0f}, {{"type", "a"}}); // Node 0
    hnsw_graph.insert({0.1f, 0.1f}, {{"type", "b"}}); // Node 1
    hnsw_graph.insert({0.2f, 0.2f}, {{"type", "a"}}); // Node 2
    hnsw_graph.insert({0.3f, 0.3f}, {{"type", "c"}}); // Node 3

    // Filter for type "a"
    auto filter_a = [](const hnsw::Metadata& meta) {
        auto it = meta.find("type");
        return it != meta.end() && it->second == "a";
    };

    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors({0.0f, 0.0f}, 2, filter_a);
    assert(results.size() == 2);
    std::vector<int> ids;
    for(const auto& r : results) {
        ids.push_back(r.id);
    }
    std::sort(ids.begin(), ids.end());
    assert(ids[0] == 0);
    assert(ids[1] == 2);

    // Filter for type "b"
    auto filter_b = [](const hnsw::Metadata& meta) {
        auto it = meta.find("type");
        return it != meta.end() && it->second == "b";
    };

    results = hnsw_graph.k_nearest_neighbors({0.0f, 0.0f}, 1, filter_b);
    assert(results.size() == 1);
    assert(results[0].id == 1);

    std::cout << "test_metadata_filtering passed." << std::endl;
}

void test_data_inclusion() {
    hnsw::HNSW hnsw_graph(2, 2, 5, 5); // vector_dimension = 2

    hnsw::Metadata meta = {{"key", "value"}};
    std::vector<float> vec = {1.0f, 2.0f};
    hnsw_graph.insert(vec, meta); // Node 0

    // Test including different data types
    std::vector<hnsw::QueryResult> results = hnsw_graph.k_nearest_neighbors({1.1f, 2.1f}, 1, nullptr, {hnsw::Include::ID});
    assert(results.size() == 1);
    assert(results[0].id == 0);
    assert(float_equals(results[0].distance, 0.0f)); // Default value
    assert(results[0].metadata.empty());
    assert(results[0].vector.empty());

    results = hnsw_graph.k_nearest_neighbors({1.1f, 2.1f}, 1, nullptr, {hnsw::Include::ID, hnsw::Include::DISTANCE});
    assert(results.size() == 1);
    assert(results[0].id == 0);
    assert(!float_equals(results[0].distance, 0.0f));
    assert(results[0].metadata.empty());
    assert(results[0].vector.empty());

    results = hnsw_graph.k_nearest_neighbors({1.1f, 2.1f}, 1, nullptr, {hnsw::Include::ID, hnsw::Include::METADATA});
    assert(results.size() == 1);
    assert(results[0].id == 0);
    assert(float_equals(results[0].distance, 0.0f));
    assert(results[0].metadata == meta);
    assert(results[0].vector.empty());

    results = hnsw_graph.k_nearest_neighbors({1.1f, 2.1f}, 1, nullptr, {hnsw::Include::ID, hnsw::Include::VECTOR});
    assert(results.size() == 1);
    assert(results[0].id == 0);
    assert(float_equals(results[0].distance, 0.0f));
    assert(results[0].metadata.empty());
    assert(results[0].vector == vec);

    std::cout << "test_data_inclusion passed." << std::endl;
}

void test_database_save_load() {
    std::string db_path = "test_db.bin";
    {
        hnsw::Database db(db_path, 2);
        db.insert({1.0f, 2.0f}, {{"type", "a"}});
        db.insert({3.0f, 4.0f}, {{"type", "b"}});
        db.save();
    }

    {
        hnsw::Database db(db_path, 2, 16, 200, 50, hnsw::DistanceMetric::L2, true); // Read only
        std::vector<hnsw::QueryResult> results = db.query({1.1f, 2.1f}, 1, nullptr, {hnsw::Include::ID, hnsw::Include::METADATA});
        assert(results.size() == 1);
        assert(results[0].id == 0);
        assert(results[0].metadata["type"] == "a");
    }
    std::remove(db_path.c_str());
    std::cout << "test_database_save_load passed." << std::endl;
}

int main() {
    test_l2_distance_hnsw();
    test_cosine_distance_hnsw();
    test_inner_product_distance_hnsw();
    test_node_structure();
    test_vector_storage();
    test_search_layer();
    test_full_hnsw_insertion();
    test_k_nearest_neighbors();
    test_vector_dimension_enforcement();
    test_metadata_filtering();
    test_data_inclusion();
    test_database_save_load();

    std::cout << "All tests passed!" << std::endl;

    return 0;
}
